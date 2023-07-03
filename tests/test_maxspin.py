
from maxspin import spatial_information, pairwise_spatial_information
from maxspin.binning import spatial_neighbors

import numpy as np
from numpy.random import Generator, default_rng
from anndata import AnnData

NCELLS = 1000
NGENES = 10
NEPOCHS = 100
NEVALSAMPLES = 100


def rand_spatial(rng: Generator, ncells: int):
    return rng.uniform(low=-1.0, high=1.0, size=(ncells, 2))


def rand_poisson_adata(rng: Generator, ncells: int, ngenes: int):
    λ = rng.gamma(shape=1.0, scale=1.0, size=(ncells, ngenes))
    X = rng.poisson(lam=λ)

    adata = AnnData(
        obsm={"spatial": rand_spatial(rng, ncells)},
        X=X,
        dtype=np.int32
    )
    spatial_neighbors(adata)
    return adata

def rand_normal_adata(rng: Generator, ncells: int, ngenes: int):
    X = rng.normal(loc=0.0, scale=1.0, size=(ncells, ngenes))

    adata =  AnnData(
        obsm={"spatial": rand_spatial(rng, ncells)},
        X=X,
        layers={"std": rng.gamma(shape=1.0, scale=1.0, size=(ncells, ngenes))},
        dtype=np.float32
    )
    spatial_neighbors(adata)
    return adata

def check_spatial_information_output(adata: AnnData):
    assert("spatial_information" in adata.var)
    assert(np.isfinite(adata.var.spatial_information).all())

    assert("spatial_information_pvalue" in adata.var)
    assert(np.isfinite(adata.var.spatial_information_pvalue).all())

    assert("spatial_information_log_pvalue" in adata.var)
    assert(np.isfinite(adata.var.spatial_information_log_pvalue).all())

    assert("spatial_information_acc" in adata.layers)
    assert(np.isfinite(adata.layers["spatial_information_acc"]).all())

def test_poisson_spatial_information():
    adata = rand_poisson_adata(default_rng(0), NCELLS, NGENES)
    spatial_information(
        adata, prior="gamma",
        nepochs=NEPOCHS, nevalsamples=NEVALSAMPLES)
    check_spatial_information_output(adata)

def test_normal_spatial_information():
    adata = rand_normal_adata(default_rng(0), NCELLS, NGENES)
    spatial_information(
        adata, prior="gaussian",
        nepochs=NEPOCHS, nevalsamples=NEVALSAMPLES)
    check_spatial_information_output(adata)

def check_pairwise_spatial_information_output(adata: AnnData):
    assert("pairwise_spatial_information" in adata.varp)
    assert(np.isfinite(adata.varp["pairwise_spatial_information"]).all())

def test_poisson_pairwise_spatial_information():
    adata = rand_poisson_adata(default_rng(0), NCELLS, NGENES)
    pairwise_spatial_information(
        adata, prior="gamma",
        nepochs=NEPOCHS, nevalsamples=NEVALSAMPLES)
    check_pairwise_spatial_information_output(adata)

def test_normal_pairwise_spatial_information():
    adata = rand_normal_adata(default_rng(0), NCELLS, NGENES)
    pairwise_spatial_information(
        adata, prior="gaussian",
        nepochs=NEPOCHS, nevalsamples=NEVALSAMPLES)
    check_pairwise_spatial_information_output(adata)
