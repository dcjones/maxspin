
# Maxspin

Maxspin (maximization of spatial information) is an information theoretic
approach to quantifying the degree of spatial organization in spatial
transcriptomics (or other spatial omics) data.

## Installation

The python package can be installed with:
```sh
pip install https://github.com/dcjones/maxspin/tarball/main
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

spatial_information(adata)
```

This adds a `spatial_information` column to the `var` metadata.

Similarly, pairwise spatial information can be computed with
`pairwise_spatial_information`. This function will test every pair of genes,
which is pretty impractical for large numbers of genes, so it's a good idea to
subset the `AnnData` object before calling this.


```python
from maxspin import pairwise_spatial_information

pairwise_spatial_information(adata)
```

## Interpreting the spatial information score

The method compute a score for every cell/spot that's in `[0,1]`, like a
correlation but typically much smaller, and sums them to arrive at a spatial
information score that is then in `[0, ncells]`. It's possible to normalize for
the number of cells by just dividing, but by default a pattern involving more
cells is considered more spatially coherent, hence the sum.

## Normalization

There are different ways spatial information can be computed. By default, no
normalization is done and spatial information is computed on absolute counts.
Uncertainty is incorporated using a Gamma-Poisson model. Whether this is
appropriate or not depends on the platform. Often the total counts per cell or
spot is a confounding factor that should be normalized out.

I developed the normalization method [vanity](https://github.com/dcjones/vanity)
in part as convenient way to normalize spatial transcriptomics data in a way
that provides uncertainty estimates. The preferred way of running vanity + maxspin is then:

```python
from maxspin import spatial_information
from vanity import normalize_vanity

normalize_vanity(adata)
spatial_information(adata, prior="normal")

```

If you'd like to avoid the trouble of using uncertainty estimates, and use more
standard normalization methods, the simplest thing to do right now is to set the
standard deviation to some small value, like so
```python
from maxspin import spatial_information
import scanpy as sc
import numpy as np

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

adata.obsm["std"] = np.full(adata.shape, 1e-6)
spatial_information(adata_norm, prior="normal")

```

