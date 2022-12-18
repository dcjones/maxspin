
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
import h5py
import time

from .objectives import genewise_js
from .binning import spatially_bin_adata

Array = Any

def spatial_information(
        adatas: Union[AnnData, list[AnnData]],
        layer: Optional[str]=None,
        nwalksteps: int=1,
        stepsize: int=5,
        lr: float=1e-2,
        nepochs: int=8000,
        binsizes: List[int]=[4, 8, 16],
        binweights: Union[Callable, List[float]]=lambda binsize: np.sqrt(binsize),
        max_unimproved_count: Optional[int]=50,
        seed: int=0,
        prior: Optional[str]="gamma",
        std_layer: str="std",
        prior_k: float=0.01,
        prior_theta: float=1.0,
        prior_a: float=1.0,
        estimate_scales: bool=False,
        chunk_size: Optional[int]=None,
        alpha_layer: str="alt",
        beta_layer: str="ref",
        receiver_signals: Optional[Any]=None,
        resample_frequency: int=10,
        nevalsamples: int=1000,
        preload: bool=True,
        quiet: bool=False):
    """Compute spatial information for each gene in an an `AnnData`, or list of `AnnData`
    objects.

    If a list of of `AnnData` objects is used, the spatial information score is
    computed jointly across each.

    Every adata must have a spatial neighborhood graph provided. The easiest way
    to do this is with:
        `squidpy.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")`

    Spatial information scores, which are added to the
    `adata.var["spatial_information"]`, represent a lower bound on spatial
    auto-mutual information. They are normalized so that 0 represents a total
    lack of spatial coherence, and increasing positive numbers more spatial
    coherence.

    The bound on mutual information is computed by training extremely simple
    classifiers on pairs of nearby cells/spots. The classifier is trained to
    recognize when spatial arrangement has been shuffled. Informally, spatial
    information is then defined as how easy it is to tell when expression have
    been shuffled across spatial positions. In a highly spatially coherent
    expression pattern, the distribution of pairs of nearby values shifts
    dramatically. In the lack of any spatial coherence, this distribution does
    not change.

    Nearby pairs of nodes are sampled by performing random walks on the
    neighborhood graph. The length of the walks partially controls the scale of
    the spatial patterns that are detected. Longer walks will tend to recognize
    only very broad spatial patterns, while short walks only very precise ones.
    In that way, spatial information in not necessarily comparable when two
    different walk lengths are used.

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
            a model appropriate for absolute counts, if "dirichlet" use a model
            appropriate for proprotional counts. If set to "beta" in conjunction
            with setting `alpha_layer` and `beta_layer` to two separate count layers,
            use a model suitable for testing for spatial patterns in allelic balance.
            If "gaussian", the model expects a matrix nammed `std` in `layers` holding
            standard deviations for the estimates held in `X`.
        std_layer: Name of layer containing standard deviation estimates for
            the expression values in X. Should be the same shape as X.
        prior_k: Set the `k` in a `Gamma(k, θ)` if prior is "gamma",
        prior_theta: Set the `θ` in `Gamma(k, θ)` if prior is "gamma",
        prior_a: Use either a `Beta(a, a)` or `Dirichlet(a, a, a, ...)` prior
            depending on the value of `prior`.
        estimate_scales: When using a dirichlet prior, try to estimate scales
            to adjust proportions to capture relative absolute abundance
        chunk_size: How many genes to score at a time, mainly affecting memory
            usage. When None, a reasonable number will be chosen to avoid using too much
            memory.
        alpha_layer: When using a beta prior, gives the name of the layer (in `adata.layers`)
            for the α parameter.
        beta_layer: When using a beta prior, gives the name of the layer (in `adata.layers`)
            for the β parameter.
        receiver_signals: Instead of computing auto-spatial mutual information
            for each gene, compute the spatial mutual information between
            each gene and the given signal, which must be either 1-dimensional
            or the same number of dimensions as there are genes. This signal is
            not resampled during training.
        resample_frequency: Resample expression values after this many iterations.
            This tends to be computationally expensive, so is not done every iteration
            during optimization.
        preload: If multiple AnnData objects are used, load everything into
            GPU memory at once, rather than as needed. This is considerably faster
            when GPU memory is not an issue.
        quiet: Don't print stuff to stdout while running.

    Returns:
        Modifies each `adata` with with the following keys:

        - `anndata.AnnData.var["spatial_information"]`: Lower bound on spatial
            information for each gene.
        - `anndata.AnnData.layers["spatial_information_acc"]`: Per spot/cell
            classifier accuracy. Useful for visualizing what regions were
            inferred to have high spatial coherence..
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

    Ps = [neighbor_transition_matrix(adata, self_edges=True) for adata in adatas]
    random_walk_graphs = [random_walk_matrix(P, nc, stepsize) for (nc, P) in zip(ncs, Ps)]

    # Try to calibrate chunk_size to not blow out GPU memory. Setting it here
    # to use about 1GB
    if chunk_size is None:
        chunk_size = min(ngenes, max(1, int(1e8 / 4 / max(ncs))))
    quiet or print(f"chunk size: {chunk_size}")

    if prior is not None and prior not in ["gamma", "beta", "dirichlet", "gaussian"]:
        raise Exception("Supported prior types are None, \"gamma\", \"beta\", \"dirichlet\", or \"gaussian\"")

    assert receiver_signals is None or len(receiver_signals) == nsamples

    # Right now this only works if the receiver signal has dimension 1 so
    # it broadcasts across the genes. We could in principle do some sort of
    # cartesian product, but that could easy blow up if we're not careful.
    if receiver_signals is not None:
        for receiver_signal in receiver_signals:
            assert receiver_signal.shape[1] == 1 or receiver_signal.shape[1] == ngenes

    if receiver_signals is not None:
        sample_v = lambda key, u, i: receiver_signals[i]
    else:
        sample_v = lambda key, u, i: u

    scales = None
    if prior == "dirichlet":
        if estimate_scales:
            scales = estimate_scale_factors(us, prior_a)
        else:
            scales = [np.ones((u.shape[0], 1), dtype=np.float32) for u in us]

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

    # Doesn't seem to make a difference, but hypothetically could be more stable
    # when optimizing over a collection of very different datasets.
    optimizer = optax.MultiSteps(
        optax.adam(learning_rate=lr),
        every_k_schedule=len(adatas))

    # optimizer = optax.adam(learning_rate=lr)

    train_step = make_train_step(optimizer)

    scores_chunks = []
    tpr_chunks = []
    for gene_from in range(0, ngenes, chunk_size):
        gene_to = min(gene_from + chunk_size, ngenes)
        # quiet or print(f"Scoring gene {gene_from} to gene {gene_to-1}")

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
            scales=scales,
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

    scores = jnp.concatenate(scores_chunks)
    tpr = jnp.concatenate(tpr_chunks, axis=1)

    for adata in adatas:
        adata.var["spatial_information"] = np.array(scores)

        # This is mostly monotonic, and arguably more interpretable
        # adata.var["spatial_information"] = np.mean(np.array(tpr), axis=0)

    adatas[0].layers["spatial_information_acc"] = np.array(tpr)


def score_chunk(
        xys: List[Any],
        us: List[Any],
        cell_counts: List[Any],
        objective_weights: List[Any],
        sample_v: Optional[Callable],
        u_index: Optional[Array],
        v_index: Optional[Array],
        random_walk_graphs: List[Any],
        prior: Optional[str],
        scales: Optional[List[Any]],
        αs: Optional[List[Any]],
        βs: Optional[List[Any]],
        σs: Optional[List[Any]],
        desc: str,
        prior_k: float,
        prior_θ: float,
        nwalksteps: int,
        seed: int,
        objective: Callable,
        optimizer: optax.GradientTransformation,
        train_step: Callable,
        classifier: Callable,
        nepochs: int,
        resample_frequency: int,
        nevalsamples: int,
        max_unimproved_count: int,
        preload: bool,
        quiet: bool):
    """
    Helper function to compute information scores for some subset of the genes.
    """

    assert (u_index is None and v_index is None) or (u_index.shape == v_index.shape)

    nsamples = len(us)
    ncs = [u.shape[0] for u in us]
    ngenes = us[0].shape[1] if u_index is None else u_index.shape[0]

    us_samples = [None for _ in range(nsamples)]
    vs_samples = [None for _ in range(nsamples)]

    key = jax.random.PRNGKey(seed)

    modelargs = FrozenDict({
        "nsamples": nsamples,
        "objective": objective,
        "classifier": classifier,
    })

    key, init_key = jax.random.split(key)

    vars = MINE(training=True, **modelargs).init(
        init_key,
        key,
        0, xys[0][:,0],
        jax.device_put(us[0] if u_index is None else us[0][:,u_index]),
        jax.device_put(us[0] if u_index is None else us[0][:,u_index]),
        jnp.arange(us[0].shape[0]),
        objective_weights[0])


    model_state, params = vars.pop("params")
    opt_state = optimizer.init(params)

    if preload:
        random_walk_graphs = jax.device_put(random_walk_graphs)
        us = jax.device_put(us)

        if σs is not None:
            σs = [jax.device_put(σ) for σ in σs]

        if αs is not None:
            αs = [jax.device_put(α) for α in αs]

        if βs is not None:
            βs = [jax.device_put(β) for β in βs]

        if scales is not None:
            scales = [jax.device_put(scale) for scale in scales]

    # compute means and stds over point estimates so we can shift and scale
    # to make training a little easier.
    post_θ = prior_θ/(prior_θ+1)

    u_means = []
    u_stds = []
    for (i, u) in enumerate(us):
        if prior == "dirichlet":
            p = jnp.expand_dims(αs[i] / (αs[i] + βs[i]), axis=-1)
            u_est = p * (u / jnp.sum(u, axis=1, keepdims=True))
            u_est /= scales[i]
            u_est = jnp.log1p(1e6 * u_est)
        elif prior == "gamma":
            u_est = jnp.log1p((u+prior_k*cell_counts[i]) * post_θ / cell_counts[i])
        else:
            u_est = u / cell_counts[i]

        u_means.append(jnp.mean(u_est, axis=0))
        u_stds.append(jnp.std(u_est, axis=0) + 1e-1)

    best_mi_bounds = jnp.full(ngenes, -jnp.inf)
    unimproved_count = jnp.zeros(ngenes, dtype=int)

    # Resample `us_samples[i]` signals (i.e. gene expression typically). This
    # is to account for uncertainty in actual expression while training.
    def resample_signals(i):
        u = jax.device_put(us[i])

        if prior == "dirichlet":
            us_samples[i] = sample_signals_dirichlet(
                step_key, u, αs[i], βs[i], u_means[i], u_stds[i], scales[i])
        elif prior == "gamma":
            us_samples[i] = sample_signals_gamma(
                step_key, u, u_means[i], u_stds[i], post_θ, prior_k, cell_counts[i])
        elif prior == "beta":
            us_samples[i] = sample_signals_beta(
                step_key, αs[i], βs[i])
        elif prior == "gaussian":
            us_samples[i] = sample_signals_gaussian(
                step_key, u, σs[i], u_means[i], u_stds[i], cell_counts[i])
        else:
            us_samples[i] = (u - u_means[i]) / u_stds[i]

        vs_samples[i] = sample_v(step_key, us_samples[i], i)

    # training loop
    prog = None if quiet else tqdm(total=nepochs, desc=desc)
    prog_update_freq = 50

    # debug_output = h5py.File("samples.h5", "w")

    for epoch in range(nepochs):
        mi_lower_bounds_sum = jnp.zeros(ngenes, dtype=jnp.float32)
        for i in range(nsamples):
            key, step_key = jax.random.split(key)

            receivers, receivers_logits = random_walk_graphs[i]

            walk_receivers = weighted_random_walk(
                nwalksteps, step_key, jax.device_put(receivers),
                jax.device_put(receivers_logits))

            distances = random_walk_distances(xys[i], walk_receivers)

            # print((jnp.min(distances), jnp.max(distances)))

            if epoch % resample_frequency == 0:
                resample_signals(i)

            model_state, params, opt_state, mi_lower_bounds, metrics = train_step(
                modelargs,
                cell_counts[i], distances,
                us_samples[i] if u_index is None else us_samples[i][:,u_index],
                vs_samples[i] if v_index is None else vs_samples[i][:,v_index],
                walk_receivers, objective_weights[i],
                model_state, params, opt_state, step_key)

            mi_lower_bounds_sum += mi_lower_bounds

            # Diagnostics
            # debug_output.create_dataset(f"senders_{i}", data=np.array(us_samples[i]))
            # debug_output.create_dataset(f"receivers_{i}", data=np.array(us_samples[i][walk_receivers,:]))
            # debug_output.create_dataset(f"distances_{i}", data=np.array(distances))

        unimproved_count += 1
        unimproved_count = unimproved_count.at[mi_lower_bounds_sum > best_mi_bounds].set(0)
        best_mi_bounds = jnp.maximum(best_mi_bounds, mi_lower_bounds_sum)

        # With a large number of genes, this is unlikely to be triggered
        # because we will randomly get some tiny improvement. Is there a less
        # conservative stopping criteria?
        if jnp.min(unimproved_count) > max_unimproved_count:
            break

        if prog is not None and (epoch + 1) % prog_update_freq == 0:
            prog.update(prog_update_freq)
            prog.set_postfix(mean_mi_bound=jnp.float32(jnp.mean(mi_lower_bounds)))

        # if (epoch + 1) % 500 == 0:
        #     quiet or print(f"epoch: {epoch+1}, min unimproved count: {jnp.min(unimproved_count)}, mi bound: {jnp.float32(jnp.mean(mi_lower_bounds))}")

    if prog is not None:
        prog.close()
    if not quiet and jnp.min(unimproved_count) > max_unimproved_count:
        quiet or print("Loss plateaued. Quitting early.")

    # evaluation loop
    mi_lower_bounds_sum = jnp.zeros(ngenes, dtype=jnp.float32)
    vars = {"params": params, **model_state}
    tpr = jnp.zeros((ncs[0], ngenes)) # cell/gene-wise true positive rate for the first adata

    for epoch in range(nevalsamples):
        for i in range(nsamples):
            key, step_key = jax.random.split(key)

            receivers, receivers_logits = random_walk_graphs[i]

            walk_receivers = weighted_random_walk(
                nwalksteps, step_key, jax.device_put(receivers),
                jax.device_put(receivers_logits))

            distances = jnp.sqrt(jnp.sum(jnp.square(xys[i] - xys[i][walk_receivers,:]), axis=1))

            if epoch % resample_frequency == 0:
                resample_signals(i)

            mi_lower_bounds, tpr_i = eval_step(
                modelargs,
                vars,
                step_key,
                cell_counts[i], distances,
                us_samples[i] if u_index is None else us_samples[i][:,u_index],
                vs_samples[i] if v_index is None else vs_samples[i][:,v_index],
                walk_receivers, objective_weights[i])

            mi_lower_bounds_sum += mi_lower_bounds
            if i == 0:
                tpr += tpr_i

    mi_lower_bounds_sum /= nevalsamples
    tpr /= nevalsamples

    return mi_lower_bounds_sum, tpr


def neighbor_transition_matrix(adata: AnnData, self_edges: bool=True):
    """
    Build a graph transition matrix with equal porability to each neighbor.
    """

    A = adata.obsp["spatial_connectivities"].tocoo()
    A = (A + A.transpose() + sparse.identity(A.shape[0]))

    A.data[:] = 1

    # delete self-edges when there is more than one
    if not self_edges:
        for i in np.arange(A.shape[0])[np.asarray(A.sum(axis=0)).flatten() > 1]:
            A[i,i] = 0

    P = A.multiply(1/A.sum(axis=1))
    return P


def random_walk_matrix(P: sparse.coo.coo_matrix, n: int, stepsize: int):
    """
    Construct a transition matrix for a `stepsize` random walk from each node. Each
    node has a row of destination nodes in one matrix and transition probability
    logits in the other, suitable for doing random walks efficiently.
    """

    Pk = sparse.identity(n)
    for _ in range(stepsize):
        Pk = Pk.dot(P)
    Pk = Pk.tocsr()

    nreceivers = Pk.indptr[1:] - Pk.indptr[:-1]
    max_receivers = np.max(nreceivers)
    ncells = Pk.shape[0]

    receivers = np.full([Pk.shape[0], max_receivers], -1, dtype=int)
    receiver_logits = np.full([Pk.shape[0], max_receivers], -np.inf, dtype=np.float32)

    for j in range(ncells):
        k0 = Pk.indptr[j]
        k1 = Pk.indptr[j+1]
        receivers[j,0:nreceivers[j]] = Pk.indices[k0:k1]
        receiver_logits[j,0:nreceivers[j]] = np.log(Pk.data[k0:k1])

    return (receivers, receiver_logits)


@jax.jit
def random_walk_distances(xys, walk_receivers):
    return jnp.sqrt(jnp.sum(jnp.square(xys - xys[walk_receivers,:]), axis=1))


@partial(jax.jit, static_argnums=(0,))
def weighted_random_walk(nsteps, key, receivers, receivers_logits):
    """
    Send every node on a random walk of length `nsteps` where `receivers` encodes
    edges, and `receiver_logits` are log probabilities for each edge.
    """

    senders = jnp.arange(receivers.shape[0])
    for _ in range(nsteps):
        walk_key, key = jax.random.split(key)
        senders = receivers[
            senders, jax.random.categorical(walk_key, receivers_logits[senders,:])]
    return senders


def estimate_scale_factors(us: list, prior_a: float):
    """
    When a dirichlet prior is used to model proportions, we typically are
    interested in variations in absolute abudance. This function computes
    estimates of proportional scaling factors needed to capture there absolute
    abundance changes. It does this my minimizing overall change in expression,
    an assumption that can be very wrong in some settings (i.e., when the
    overall amount of mRNA being produce changes dramatically).
    """

    scales = []
    for u in us:
        u = u + prior_a
        scales_i = np.mean(u / np.exp(np.mean(np.log(u), axis=0, keepdims=True)), axis=1)
        scales.append(np.expand_dims(scales_i, -1))
    return scales


def check_same_genes(adatas: list[AnnData]):
    """
    Make sure each AnnData in a list has the same set of genes.
    """

    for adata in adatas[1:]:
        if adata.var_names != adatas[0].var_names:
            raise Exception("AnnData objects must have the same set of genes")



class GeneNodePairClassifier(nn.Module):
    """
    Classifier on pairs of nodes.
    """
    training: bool

    @nn.compact
    def __call__(self, walk_start, walk_end):
        ncells, ngenes = walk_start.shape
        penalty = 0.0

        shift = self.param(
            "shift",
            lambda key, shape: jnp.full(shape, 0.0, dtype=jnp.float32),
            (1, ngenes))

        walk_start += shift
        walk_end += shift

        w_diff = -nn.softplus(self.param(
            "diff_weight",
            lambda key, shape: jnp.full(shape, -4.0),
            (1, ngenes)))

        w_sum = nn.softplus(self.param(
            "sum_weight",
            lambda key, shape: jnp.full(shape, -4.0),
            (1, ngenes)))

        w_prod = nn.softplus(self.param(
            "prod_weight",
            lambda key, shape: jnp.full(shape, -4.0, dtype=jnp.float32),
            (1, ngenes)))

        b = self.param(
            "bias",
            nn.initializers.zeros,
            (1, ngenes))

        score = b + \
            w_prod * walk_start * walk_end + \
            w_diff * jnp.abs(walk_start - walk_end) + \
            w_sum * jnp.abs(walk_start + walk_end)

        return score, penalty



class MINE(nn.Module):
    """
    Mutual information neural estimation. Shuffle the node signals, and train a
    classifier to distinguish shuffled from unshuffled, bounding mutual
    information.
    """

    training: bool
    nsamples: int
    objective: Callable
    classifier: Callable

    @nn.compact
    def __call__(self, key, cell_count, distances, u, v, walk_receivers, objective_weights):
        ncells, ngenes = u.shape

        # intentionally using the same key to get the same permutation here
        u_perm = jax.random.permutation(key, u)
        v_perm = jax.random.permutation(key, v)

        fe = self.classifier(training=self.training)

        score, penalty = fe(u, v[walk_receivers])
        perm_score, _ = fe(u_perm, v_perm[walk_receivers])

        # weighting scores by distance of the sampled neighbors, and excluding
        # walks that end up where they started.
        distance_falloff = nn.softplus(self.param(
            "distance_falloff",
            lambda key, shape: jnp.full(shape, 1.0, dtype=jnp.float32),
            ngenes))
        distance_falloff = jnp.expand_dims(distance_falloff, 0)
        nonzero_distance = jnp.expand_dims(distances > 0.0, 1)
        distance_weight = jnp.exp(-jnp.expand_dims(distances, 1)/distance_falloff) * nonzero_distance

        score *= distance_weight
        perm_score *= distance_weight

        mi_bounds = self.objective(score, perm_score) * objective_weights

        metrics = {"tp": nn.sigmoid(score - perm_score)}

        return mi_bounds - penalty, metrics


# Copying an idiom from: https://github.com/deepmind/optax/issues/197#issuecomment-974505149
def make_train_step(optimizer):
    @partial(jax.jit, static_argnums=(0,))
    def train_step(modelargs, cell_count, distances, u, v, walk_receivers, objective_weights, model_state, params, opt_state, key):
        def loss_fn(params):
            vars = {"params": params, **model_state}
            (mi_lower_bounds, metrics), new_model_state = MINE(
                training=True, **modelargs).apply(
                    vars, key, cell_count, distances, u, v, walk_receivers, objective_weights, mutable=["batch_stats"])
            return -jnp.mean(mi_lower_bounds), (mi_lower_bounds, metrics, new_model_state)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (neg_mean_mi_lower_bounds, (mi_lower_bounds, metrics, model_state)), grads = grad_fn(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return model_state, params, opt_state, mi_lower_bounds, metrics

    return train_step


@partial(jax.jit, static_argnums=(0,))
def eval_step(modelargs, vars, key, cell_count, distances, v, u, walk_receivers, objective_weights):
    (mi_lower_bounds, metrics), new_model_state = MINE(training=False, **modelargs).apply(
        vars, key,
        cell_count, distances, v, u,
        walk_receivers,
        objective_weights,
        mutable=["batch_stats"])

    return mi_lower_bounds, metrics["tp"]


@jax.jit
def sample_signals_gamma(key, v, v_mean, v_std, post_θ, prior_k, cell_count):
    v_sample = jnp.log1p(post_θ * jax.random.gamma(key, v + prior_k * cell_count) / cell_count)
    v_sample = (v_sample - v_mean) / v_std
    return v_sample


@jax.jit
def sample_signals_dirichlet(key, v, α, β, v_mean, v_std, scale):
    p = jnp.expand_dims(jax.random.beta(key, α, β), axis=-1)
    v_sample = jnp.log1p(1e6 * p * jax.random.dirichlet(key, v) / scale)
    v_sample = (v_sample - v_mean) / v_std
    return v_sample


@jax.jit
def sample_signals_beta(key, α, β):
    return jnp.log(jax.random.beta(key, α, β))


@jax.jit
def sample_signals_gaussian(key, μ, σ, v_mean, v_std, cell_count):
    x = (jax.random.normal(key, μ.shape) * σ + μ) / cell_count
    return (x - v_mean) / v_std