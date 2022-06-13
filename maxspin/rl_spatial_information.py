
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

from .objectives import genewise_js
from .spatial_information import \
    score_chunk, check_same_genes, neighbor_transition_matrix, \
    random_walk_matrix, estimate_scale_factors, make_train_step

def rl_spatial_information(
        adatas: Union[AnnData, list[AnnData]],
        pairs: Union[List[tuple[str, str]], List[tuple[int, int]]],
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
        resample_frequency: int=10,
        nevalsamples: int=1000,
        preload: bool=True,
        quiet: bool=False):
    """Compute the spatial information between pairs of genes.

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

    ncs = [adata.shape[0] for adata in adatas]
    quiet or print(f"ncells: {ncs}")

    if max_unimproved_count is None:
        max_unimproved_count = nepochs

    us = [adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray() for adata in adatas]

    Ps = [neighbor_transition_matrix(adata) for adata in adatas]
    random_walk_graphs = [random_walk_matrix(P, nc, stepsize) for (nc, P) in zip(ncs, Ps)]

    gene_names = list(adatas[0].var_names)

    def gene_index(i_):
        if isinstance(i_, str):
            try:
                i = gene_names.index(i_)
            except ValueError:
                raise Exception(f"Gene name {i_} not in data set")
        else:
            i = i_
        assert 0 <= i and i < ngenes
        return i

    used_genes = set()
    u_index = []
    v_index = []
    for (a, b) in pairs:
        i, j = gene_index(a), gene_index(b)
        u_index.append(i)
        used_genes.add(i)
        v_index.append(j)
        used_genes.add(j)
    used_genes = list(used_genes)

    quiet or print(f"genes in pairs: {len(used_genes)}")

    if prior is not None and prior not in ["gamma", "beta", "dirichlet"]:
        raise Exception("Supported prior types are None, \"gamma\" or \"dirichlet\"")

    sample_v = lambda key, u, i: u

    scales = None
    if prior == "dirichlet":
        scales = estimate_scale_factors(us, prior_a)

    us_chunk = [u[:,used_genes] for u in us]
    u_chunk_index = jnp.array([used_genes.index(i) for i in u_index], dtype=int)
    v_chunk_index = jnp.array([used_genes.index(j) for j in v_index], dtype=int)

    optimizer = optax.adam(learning_rate=lr)
    train_step = make_train_step(optimizer)

    if prior == "dirichlet":
        us_chunk = [u + prior_a for u in us_chunk]
        αs_chunk = [np.sum(u, axis=1) for u in us_chunk]
        βs_chunk = [np.sum(u + prior_a, axis=1) - α for (α, u) in zip(αs_chunk, us)]
    else:
        αs_chunk = None
        βs_chunk = None

    scores_chunk, tpr_chunk = score_chunk(
        us=us_chunk,
        sample_v=sample_v,
        u_index=u_chunk_index,
        v_index=v_chunk_index,
        random_walk_graphs=random_walk_graphs,
        prior=prior,
        scales=scales,
        αs=αs_chunk,
        βs=βs_chunk,
        desc=f"Scoring {len(pairs)} gene pairs",
        prior_k=prior_k,
        prior_θ=prior_theta,
        nwalksteps=nwalksteps,
        seed=seed,
        objective=genewise_js,
        optimizer=optimizer,
        train_step=train_step,
        nepochs=nepochs,
        resample_frequency=resample_frequency,
        nevalsamples=nevalsamples,
        max_unimproved_count=max_unimproved_count,
        preload=preload,
        quiet=quiet)

    gene_pair_a = [gene_names[i] for i in u_index]
    gene_pair_b = [gene_names[j] for j in v_index]
    for adata in adatas:
        adata.uns["pair_spatial_information_gene1"] = gene_pair_a
        adata.uns["pair_spatial_information_gene2"] = gene_pair_b
        adata.uns["pair_spatial_information"] = np.array(scores_chunk)
        adata.uns["pair_spatial_information_acc"] = np.array(tpr_chunk)

