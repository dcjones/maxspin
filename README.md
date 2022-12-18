
![Maxspin](https://raw.github.com/dcjones/maxspin/main/logo.png)


Maxspin (maximization of spatial information) is an information theoretic
approach to quantifying the degree of spatial organization in spatial
transcriptomics (or other spatial omics) data.

## Installation

The python package can be installed with:
```sh
pip install maxspin
```

## Basic Usage

This package operates on `AnnData` objects from the [anndata](https://github.com/scverse/anndata) package.

We assume the existence of a spatial neighborhood graph. A simple and effective
way of doing this is Delaunay triangulation, for example using [squidpy](https://github.com/scverse/squidpy).

```python
import squidpy as sq

sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
```

Spatial information can then be measured using the `spatial_information` function.

```python
from maxspin import spatial_information

spatial_information(adata, prior=None)
```

This adds a `spatial_information` column to the `var` metadata.

Similarly, pairwise spatial information can be computed with
`pairwise_spatial_information`. This function will test every pair of genes,
which is pretty impractical for large numbers of genes, so it's a good idea to
subset the `AnnData` object before calling this.


```python
from maxspin import pairwise_spatial_information

pairwise_spatial_information(adata, prior=None)
```

For a more detailed example, check out the [tutorial](https://github.com/dcjones/maxspin/blob/main/tutorial.ipynb).

## Interpreting the spatial information score

The method compute a score for every cell/spot that's in `[0,1]`, like a
correlation but typically much smaller, and sums them to arrive at a spatial
information score that is then in `[0, ncells]`. It's possible to normalize for
the number of cells by just dividing, but by default a pattern involving more
cells is considered more spatially coherent, hence the sum.

## Normalization

There are different ways spatial information can be computed. By default, no
normalization is done and spatial information is computed on absolute counts.
Uncertainty is incorporated using a Gamma-Poisson model.

If `prior=None` is used, the method makes no attempt to account for estimation
uncertainty and computes spatial information directly on whatever is in
`adata.X`.

The recommended way to run `spatial_information` is with some kind of normalized
estimate of expression with some uncertainty estimation. There are two
recommended ways of doing this: SCVI and Vanity.


## SCVI

[SCVI](https://scvi-tools.org/) is a convenient and versatile probabilistic
model of sequencing experiments, from which we can sample from the posterior to
get normalized point estimates with uncertainty.

Using Maxspin with SCVI looks something like this:


```python
import scvi
import numpy as np
from maxspin import spatial_information

scvi.model.SCVI.setup_anndata(adata)
model = scvi.model.SCVI(adata, n_latent=20)

# Sample log-expression values from the posterior.
posterior_samples = np.log(model.get_normalized_expression(return_numpy=True, return_mean=False, n_samples=20, library_size="latent"))
adata_scvi = adata.copy()
adata_scvi.X = np.mean(posterior_samples, axis=0)
adata_scvi.layers["std"] = np.std(posterior_samples, axis=0)

spatial_information(adata_scvi, prior="gaussian")
```

The [tutorial](https://github.com/dcjones/maxspin/blob/main/tutorial.ipynb) has
a more in depth example of using SCVI.

## Vanity


I developed the normalization method [vanity](https://github.com/dcjones/vanity)
in part as convenient way to normalize spatial transcriptomics data in a way
that provides uncertainty estimates. The preferred way of running vanity + maxspin is then:

```python
from maxspin import spatial_information
from vanity import normalize_vanity

normalize_vanity(adata)
spatial_information(adata, prior="gaussian")

```

Compared to SCVI, this model more aggressively shrinks low expression genes,
which might cause it to miss something very subtle, but is less likely to detect
spurious patterns.
