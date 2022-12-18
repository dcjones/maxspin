from anndata import AnnData
from flax import linen as nn
from functools import partial
from jax.experimental.maps import FrozenDict
from scipy import sparse
from tqdm import tqdm
from typing import Optional, Any, Callable, List, Union
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sys

from .objectives import genewise_js
from .binning import spatially_bin_adata
from .spatial_information import check_same_genes, neighbor_transition_matrix, \
    random_walk_matrix, score_chunk, make_train_step, GeneNodePairClassifier

Array = Any

def pairwise_spatial_information(
        adatas: Union[AnnData, list[AnnData]],
        layer: Optional[str]=None,
        nwalksteps: int=2,
        stepsize: int=5,
        lr: float=1e-2,
        nepochs: int=8000,
        binsizes: List[int]=[4, 8, 16],
        binweights: Union[Callable, List[float]]=lambda binsize: np.sqrt(binsize),
        nevalsamples: int=1000,
        max_unimproved_count: Optional[int]=50,
        seed: int=0,
        prior: Optional[str]="gamma",
        std_layer: str="std",
        prior_k: float=0.01,
        prior_theta: float=10.0,
        prior_a: float=1.0,
        chunk_size: Optional[int]=None,
        alpha_layer: str="alt",
        beta_layer: str="ref",
        resample_frequency: int=10,
        preload: bool=True,
        quiet: bool=False):
    """
    Compute pairwise spatial information for each pair of genes in an `AnnData`
    or list of `AnnData` objects.

    Args:
        adatas: Either a single `AnnData` objects, or a list of `AnnData` objects.
            If a list is given, they must all have the same set of genes in the
            same order. Each `AnnData` must have a spatial neighborhood graph
            provided in `obsp["spatial_connectivities"]` This can be done with
            the `squidpy.gr.spatial_neighbors` function.
        layer: Name of layer to use. If `None`, the `X` matrix is used.
        nwalksteps: Random walks take this many steps. Lengthening walks
            (by increasing this parameter or `stepsize`) will make the test less
            sensitive to smaller scale spatial variations, but more sensitive to
            large scale variations.
        stepsize: Each random walk step follows this many edges on the
            neighborhood graph.
        lr: Optimizer learning rate.
        nepochs: Run the optimization step for this many iterations.
        nevalsamples: Estimate MI bound by resampling expression and random
            walks this many times.
        max_unimproved_count: Early stopping criteria for optimization. If the
            the MI lower bound has not been improved for any gene for this many
            iterations, stop iterating.
        seed: Random number generator seed.
        prior: Account for uncertainty in expression estimates by resampling
            expression while training. If `None`, do no resampling. If "gamma", use
            a model appropriate for absolute counts. If set to "beta" in conjunction
            with setting `alpha_layer` and `beta_layer` to two separate count layers,
            use a model suitable for testing for spatial patterns in allelic balance.
            If "gaussian", the model expects a matrix nammed `std` in `obsm` holding
            standard deviations for the estimates held in `X`.
        prior_k: Set the `k` in a `Gamma(k, θ)` if prior is "gamma",
        prior_theta: Set the `θ` in `Gamma(k, θ)` if prior is "gamma",
        prior_a: Use a `Beta(a, a)` prior.
        chunk_size: How many genes to score at a time, mainly affecting memory
            usage. When None, a reasonable number will be chosen to avoid using too much
            memory.
        resample_frequency: Resample expression values after this many iterations.
            This tends to be computationally expensive, so is not done every iteration
            during optimization.
        preload: If multiple AnnData objects are used, load everything into
            GPU memory at once, rather than as needed. This is considerably faster
            when GPU memory is not an issue.
        quiet: Don't print stuff to stdout while running.


    Returns:
        Modifies each `adata` adding:

          - `anndata.AnnData.varp["pairwise_spatial_information"]`: Pairwise
             spatial information matrix between genes.
    """

    if isinstance(adatas, AnnData):
        adatas = [adatas]

    check_same_genes(adatas)
    if not all(["spatial_connectivities" in adata.obsp for adata in adatas]):
        raise Exception(
            """Every adata must have 'spatial_connectivities' set. Call, for
            example, `squidpy.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")`""")

    nsamples = len(adatas)
    quiet or print(f"nsamples: {nsamples}")

    ngenes = adatas[0].shape[1]
    quiet or print(f"ngenes: {ngenes}")

    # Binning: bin cells/spots and treat it as further observations, which
    # can boost sensitivity with sparse data
    concatenated_adatas = []
    cell_counts = []
    objective_weights = []
    for adata in adatas:
        concatenated_adatas.append(adata)
        cell_counts.append(1)
        objective_weights.append(1.0)

        binned_adatas = [spatially_bin_adata(adata, binsize, std_layer, layer=layer) for binsize in binsizes]
        concatenated_adatas.extend(binned_adatas)
        cell_counts.extend(binsizes)

        if isinstance(binweights, Callable):
            objective_weights.extend([binweights(binsize) for binsize in binsizes])
        elif isinstance(binweights, List):
            assert len(binsizes) == len(binweights)
            objective_weights.extend(binweights)

    adatas = concatenated_adatas

    print(len(adatas))

    # Find a reasonable scale for coordinates
    mean_neighbor_dist = 0.0
    total_cell_count = 0
    for adata in adatas:
        neighbors = adata.obsp["spatial_connectivities"].tocoo()
        xy = adata.obsm["spatial"]
        mean_neighbor_dist += \
            np.sum(np.sqrt(np.sum(np.square(xy[neighbors.row,:] - xy[neighbors.col,:]), axis=1)))
        total_cell_count += xy.shape[0]
    mean_neighbor_dist /= total_cell_count

    xys = []
    for adata in adatas:
        xys.append(jnp.array(adata.obsm["spatial"] / mean_neighbor_dist))

    ncs = [adata.shape[0] for adata in adatas]
    quiet or print(f"ncells: {ncs}")

    if max_unimproved_count is None:
        max_unimproved_count = nepochs

    if layer is None:
        us = [adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray() for adata in adatas]
    else:
        us = [adata.layers[layer] if isinstance(adata.layers[layer], np.ndarray) else adata.layers[layer].toarray() for adata in adatas]
    us = [u.astype(np.float32) for u in us]

    Ps = [neighbor_transition_matrix(adata) for adata in adatas]
    random_walk_graphs = [random_walk_matrix(P, nc, stepsize) for (nc, P) in zip(ncs, Ps)]

    # Try to calibrate chunk_size to not blow out GPU memory. Setting it here
    # to use about 1GB
    if chunk_size is None:
        chunk_size = min(ngenes, max(1, int(1e8 / 4 / max(ncs))))
    quiet or print(f"chunk size: {chunk_size}")

    if prior is not None and prior not in ["gamma", "beta", "dirichlet", "gaussian"]:
        raise Exception("Supported prior types are None, \"gamma\", \"beta\", \"dirichlet\", or \"gaussian\"")

    if prior == "dirichlet":
        raise Exception("The dirichlet prior is not supported for pairwise spatial information.")

    αs = []
    βs = []
    if prior == "beta":
        for adata in adatas:
            for lyr in [alpha_layer, beta_layer]:
                if lyr not in adata.layers:
                    raise Exception(f"Missing layer \"{lyr}\" needed for beta prior")
            αs.append(adata.layers[alpha_layer])
            βs.append(adata.layers[beta_layer])

    σs = []
    if prior == "gaussian":
        for adata in adatas:
            if std_layer not in adata.layers:
                raise Exception(f"Gaussian prior requires a `{std_layer}` matrix in `layers`")
            σs.append(adata.layers[std_layer])

    optimizer = optax.adam(learning_rate=lr)
    train_step = make_train_step(optimizer)

    scores = np.zeros((ngenes, ngenes), dtype=np.float32)

    for receiver_gene in range(ngenes):
        print(f"Scoring all pairs with gene {receiver_gene}")
        sample_v = lambda key, u, i: u[:,receiver_gene:receiver_gene+1]

        scores_chunks = []
        tpr_chunks = []
        for gene_from in range(0, ngenes, chunk_size):
            gene_to = min(gene_from + chunk_size, ngenes)

            us_chunk = [u[:,gene_from:gene_to] for u in us]
            αs_chunk = None
            βs_chunk = None
            σs_chunk = None

            if prior == "beta":
                αs_chunk = [α[:,gene_from:gene_to] + prior_a for α in αs]
                βs_chunk = [β[:,gene_from:gene_to] + prior_a for β in βs]
            elif prior == "dirichlet":
                for u in us_chunk:
                    u += prior_a
                αs_chunk = [np.sum(u, axis=1) for u in us_chunk]
                βs_chunk = [np.sum(u + prior_a, axis=1) - α for (α, u) in zip(αs_chunk, us)]
            elif prior == "gaussian":
                σs_chunk = [σ[:,gene_from:gene_to] for σ in σs]

            scores_chunk, tpr_chunk = score_chunk(
                xys=xys,
                us=us_chunk,
                cell_counts=cell_counts,
                objective_weights=objective_weights,
                sample_v=sample_v,
                u_index=None,
                v_index=None,
                random_walk_graphs=random_walk_graphs,
                prior=prior,
                scales=None,
                αs=αs_chunk,
                βs=βs_chunk,
                σs=σs_chunk,
                desc=f"Scoring genes {gene_from} to {gene_to}",
                prior_k=prior_k,
                prior_θ=prior_theta,
                nwalksteps=nwalksteps,
                seed=seed,
                objective=genewise_js,
                optimizer=optimizer,
                train_step=train_step,
                classifier=GeneNodePairClassifier,
                nepochs=nepochs,
                resample_frequency=resample_frequency,
                nevalsamples=nevalsamples,
                max_unimproved_count=max_unimproved_count,
                preload=preload,
                quiet=quiet)

            scores_chunks.append(scores_chunk)
            tpr_chunks.append(tpr_chunk)

        scores[:,receiver_gene] = jnp.concatenate(scores_chunks)

    for adata in adatas:
        adata.varp["pairwise_spatial_information"] = np.array(scores)



