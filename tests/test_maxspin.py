
from maxspin import spatial_information, pairwise_spatial_information

import numpy as np
from numpy.random import Generator, default_rng
from anndata import AnnData
from squidpy.gr import spatial_neighbors


def rand_spatial(rng: Generator, ncells: int):
    return rng.uniform(low=-1.0, high=1.0, size=(ncells, 2))


def rand_poisson_adata(rng: Generator, ncells: int, ngenes: int):
    λ = rng.gamma(shape=1.0, scale=1.0, size=(ncells, ngenes))
    X = rng.poisson(lam=λ)

    return AnnData(
        obsm={"spatial": rand_spatial(rng, ncells)},
        X=X,
        dtype=np.int32
    )

def rand_normal_adata(rng: Generator, ncells: int, ngenes: int):
    X = rng.normal(loc=0.0, scale=1.0, size=(ncells, ngenes))

    return AnnData(
        obsm={"spatial": rand_spatial(rng, ncells)},
        X=X,
        layers={"std": rng.gamma(shape=1.0, scale=1.0, size=(ncells, ngenes))},
        dtype=np.float32
    )

if __name__ == "__main__":
    ncells = 1000
    ngenes = 20
    rng = default_rng(0)

    data = [
        ("gaussian", rand_normal_adata(rng, ncells, ngenes)),
        ("poisson", rand_poisson_adata(rng, ncells, ngenes))
    ]

    for (prior, adata) in data:
        spatial_neighbors(adata, coord_type="generic")

        spatial_information(adata, nepochs=100, nevalsamples=100, quiet=True)
        assert("spatial_information" in adata.var)
        assert("spatial_information_pvalue" in adata.var)
        assert("spatial_information_log_pvalue" in adata.var)
        assert("spatial_information_acc" in adata.layers)

        pairwise_spatial_information(adata, nepochs=100, nevalsamples=100, quiet=True)
        assert("pairwise_spatial_information" in adata.varp)
