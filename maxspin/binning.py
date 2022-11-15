
from anndata import AnnData
import numpy as np
import squidpy as sq


def kdtree_bin_points(xy, leafsize: int, firstdim: int):
    """
    Bin spatial data using KD-tree construction, but preserving exact leaf size.
    Instead of splitting always at the median, we make sure each split remains
    divisible by the leafsize.
    """

    ncells = xy.shape[0]

    assert ncells % leafsize == 0

    groups = []

    def kdbuild(xy, idxs, dim: int=0):
        nodesize = xy.shape[0]

        if nodesize == leafsize:
            groups.append(idxs)
            return

        perm = xy[:,dim].argsort()

        xy = xy[perm,:]
        idxs = idxs[perm]

        mid = int(((nodesize / leafsize) // 2) * leafsize)

        dim = (dim + 1) % 2

        kdbuild(xy[:mid,:], idxs[:mid], dim)
        kdbuild(xy[mid:,:], idxs[mid:], dim)

    kdbuild(xy, np.arange(ncells), firstdim)

    clusters = np.zeros(ncells, dtype=int)
    for (cluster, group) in enumerate(groups):
        clusters[group] = cluster

    return clusters



def spatially_bin_adata(adata: AnnData, binsize: float, kdfirstdim: int=0):
    """
    Spatially bin an AnnData.
    """

    ncells, ngenes = adata.shape

    remainder = ncells % binsize

    # print(f"Discarding {remainder} cells to keep bin sizes equal")
    rng = np.random.default_rng()
    mask = np.ones(ncells, dtype=bool)
    mask[rng.permutation(np.arange(ncells))[0:remainder]] = 0

    adata = adata[mask,:]
    ncells, ngenes = adata.shape

    xy = np.asarray(adata.obsm["spatial"])

    clusters = kdtree_bin_points(xy, binsize, kdfirstdim)

    # TODO: may need to make this dense if its a hdf5 dataset, but don't
    # do this to a sparse arrays. Hmm.
    # X = np.asarray(adata.X)
    X = adata.X

    nclusters = np.max(clusters)+1
    binned_X = np.zeros((nclusters, ngenes), dtype=X.dtype)
    binned_xy = np.zeros((nclusters, 2), dtype="float64")
    for i in range(ncells):
        binned_X[clusters[i],:] += X[i,:]
        binned_xy[clusters[i],:] += xy[i,:]
    binned_xy /= binsize

    # make sure we didn't lose anything here
    assert np.isclose(
        np.sum(binned_X, dtype=np.float64),
        np.sum(X, dtype=np.float64),
        atol=1e-1,
        rtol=1e-8)

    # bin standard deviations if they are present
    obsm = {"spatial": binned_xy}
    if "std" in adata.obsm:
        std = adata.obsm["std"]
        binned_std = np.zeros((nclusters, ngenes))
        for i in range(ncells):
            binned_std[clusters[i],:] += np.square(std[i,:])
        binned_std = np.sqrt(binned_std)
        obsm["std"] = binned_std

    binned_adata = AnnData(
        X=binned_X,
        var=adata.var,
        obsm=obsm,
        uns={"binsize": binsize})

    sq.gr.spatial_neighbors(binned_adata, coord_type="generic", delaunay=True)

    return binned_adata

