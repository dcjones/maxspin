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
from .spatial_information import check_same_genes, neighbor_transition_matrix, \
    random_walk_matrix, score_chunk, make_train_step, GeneNodePairClassifier

Array = Any

def pairwise_spatial_information(
        adatas: Union[AnnData, list[AnnData]],
        nwalksteps: int=2,
        stepsize: int=5,
        lr: float=1e-2,
        nepochs: int=8000,
        max_unimproved_count: Optional[int]=50,
        seed: int=0,
        prior: Optional[str]="gamma",
        prior_k: float=0.1,
        prior_theta: float=10.0,
        prior_a: float=1.0,
        estimate_scales: bool=False,
        chunk_size: Optional[int]=None,
        alpha_layer: str="alt",
        beta_layer: str="ref",
        resample_frequency: int=10,
        nevalsamples: int=1000,
        preload: bool=True,
        quiet: bool=False):

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

    ncs = [adata.shape[0] for adata in adatas]
    quiet or print(f"ncells: {ncs}")

    if max_unimproved_count is None:
        max_unimproved_count = nepochs

    us = [adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray() for adata in adatas]
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
            if "std" not in adata.obsm:
                raise Exception("Gaussian prior requires a `std` matrix in `obsm`")
            σs.append(adata.obsm["std"])

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
                us=us_chunk,
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



