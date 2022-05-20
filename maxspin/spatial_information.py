
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

def spatial_information(
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
        prior_k: Set the `k` in a `Gamma(k, θ)` if prior is "gamma",
        prior_theta: Set the `θ` in `Gamma(k, θ)` if prior is "gamma",
        prior_a: Use either a `Beta(a, a)` or `Dirichlet(a, a, a, ...)` prior
            depending on the value of `prior`.
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

    ncs = [adata.shape[0] for adata in adatas]
    quiet or print(f"ncells: {ncs}")

    if max_unimproved_count is None:
        max_unimproved_count = nepochs

    us = [adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray() for adata in adatas]

    Ps = [neighbor_transition_matrix(adata) for adata in adatas]
    random_walk_graphs = [random_walk_matrix(P, nc, stepsize) for (nc, P) in zip(ncs, Ps)]

    # Try to calibrate chunk_size to not blow out GPU memory. Setting it here
    # to use about 1GB
    if chunk_size is None:
        chunk_size = max(1, int(1e8 / 4 / max(ncs)))
    quiet or print(f"chunk size: {chunk_size}")

    if prior is not None and prior not in ["gamma", "beta", "dirichlet"]:
        raise Exception("Supported prior types are None, \"gamma\", \"beta\", or \"dirichlet\"")

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
        scales = estimate_scale_factors(us, prior_a)

    αs = []
    βs = []
    if prior == "beta":
        for adata in adatas:
            for lyr in [alpha_layer, beta_layer]:
                if lyr not in adata.layers:
                    raise Exception(f"Missing layer \"{lyr}\" needed for beta prior")
            αs.append(adata.layers[alpha_layer])
            βs.append(adata.layers[beta_layer])

    optimizer = optax.adam(learning_rate=lr)
    train_step = make_train_step(optimizer)

    scores_chunks = []
    tpr_chunks = []
    for gene_from in range(0, ngenes, chunk_size):
        gene_to = min(gene_from + chunk_size, ngenes)
        # quiet or print(f"Scoring gene {gene_from} to gene {gene_to-1}")

        us_chunk = [u[:,gene_from:gene_to] for u in us]

        if prior == "beta":
            αs_chunk = [α[:,gene_from:gene_to] + prior_a for α in αs]
            βs_chunk = [β[:,gene_from:gene_to] + prior_a for β in βs]
        elif prior == "dirichlet":
            for u in us_chunk:
                u += prior_a
            αs_chunk = [np.sum(u, axis=1) for u in us_chunk]
            βs_chunk = [np.sum(u, axis=1) - α for (α, u) in zip(αs_chunk, us)]
        else:
            αs_chunk = None
            βs_chunk = None

        scores_chunk, tpr_chunk = score_chunk(
            us=us_chunk,
            sample_v=sample_v,
            random_walk_graphs=random_walk_graphs,
            prior=prior,
            scales=scales,
            αs=αs_chunk,
            βs=βs_chunk,
            desc=f"Scoring genes {gene_from} to {gene_to}",
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

        scores_chunks.append(scores_chunk)
        tpr_chunks.append(tpr_chunk)

    scores = jnp.concatenate(scores_chunks)
    tpr = jnp.concatenate(tpr_chunks, axis=1)

    for adata in adatas:
        adata.var["spatial_information"] = np.array(scores)
        adata.layers["spatial_information_acc"] = np.array(tpr)


def score_chunk(
        us: List[Any],
        sample_v: Optional[Callable],
        random_walk_graphs: List[Any],
        prior: Optional[str],
        scales: Optional[List[Any]],
        αs: Optional[List[Any]],
        βs: Optional[List[Any]],
        desc: str,
        prior_k: float,
        prior_θ: float,
        nwalksteps: int,
        seed: int,
        objective: Callable,
        optimizer: optax.GradientTransformation,
        train_step: Callable,
        nepochs: int,
        resample_frequency: int,
        nevalsamples: int,
        max_unimproved_count: int,
        preload: bool,
        quiet: bool):
    """
    Helper function to compute information scores for some subset of the genes.
    """

    nsamples = len(us)
    ncs = [u.shape[0] for u in us]
    ngenes = us[0].shape[1]

    us_samples = [None for _ in range(nsamples)]
    vs_samples = [None for _ in range(nsamples)]

    key = jax.random.PRNGKey(seed)

    modelargs = FrozenDict({
        "objective": objective,
    })

    key, init_key = jax.random.split(key)

    vars = MINE(training=True, **modelargs).init(
        init_key,
        key,
        jax.device_put(us[0]),
        jax.device_put(us[0]),
        jnp.arange(us[0].shape[0]))

    model_state, params = vars.pop("params")
    opt_state = optimizer.init(params)

    if preload:
        random_walk_graphs = jax.device_put(random_walk_graphs)
        us = jax.device_put(us)

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
            u_est = jnp.log1p((u+prior_k) * post_θ)
        else:
            u_est = u

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
                step_key, u, u_means[i], u_stds[i], post_θ, prior_k)
        elif prior == "beta":
            vs_samples[i] = sample_signals_beta(
                step_key, αs[i], βs[i])
        else:
            us_samples[i] = u

        vs_samples[i] = sample_v(step_key, us_samples[i], i)

    # training loop
    prog = None if quiet else tqdm(total=nepochs, desc=desc)
    prog_update_freq = 50

    for epoch in range(nepochs):
        mi_lower_bounds_sum = jnp.zeros(ngenes, dtype=jnp.float32)
        for i in range(nsamples):
            key, step_key = jax.random.split(key)

            receivers, receivers_logits = random_walk_graphs[i]

            walk_receivers = weighted_random_walk(
                nwalksteps, step_key, jax.device_put(receivers),
                jax.device_put(receivers_logits))

            if epoch % resample_frequency == 0:
                resample_signals(i)

            model_state, params, opt_state, mi_lower_bounds, metrics = train_step(
                modelargs,
                us_samples[i],
                vs_samples[i],
                walk_receivers,
                model_state, params, opt_state, step_key)

            mi_lower_bounds_sum += mi_lower_bounds

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

            if epoch % resample_frequency == 0:
                resample_signals(i)

            mi_lower_bounds, tpr_i = eval_step(
                modelargs,
                vars,
                step_key,
                vs_samples[i],
                us_samples[i],
                walk_receivers)

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



class GenewiseNodePairClassifier(nn.Module):
    """
    Genewise classifier on pairs of nodes.
    """
    training: bool

    @nn.compact
    def __call__(self, walk_start, walk_end):
        ngenes = walk_start.shape[1]
        penalty = 0.0

        w_diff = -nn.softplus(self.param(
            "diff_weight",
            lambda key, shape: jnp.full(shape, -4.0),
            (1, ngenes)))

        w_sum = nn.softplus(self.param(
            "sum_weight",
            lambda key, shape: jnp.full(shape, -4.0),
            (1, ngenes)))

        s_sum = self.param(
            "sum_shift",
            nn.initializers.zeros,
            (1, ngenes))

        b = self.param(
            "bias",
            nn.initializers.zeros,
            (1, ngenes))

        score = b + \
            w_diff * jnp.abs(walk_start - walk_end) + \
            w_sum * jnp.abs(walk_start + walk_end - s_sum)

        return score, penalty



class MINE(nn.Module):
    """
    Mutual information neural estimation. Shuffle the node signals, and train a
    classifier to distinguish shuffled from unshuffled, bounding mutual
    information.
    """

    training: bool
    objective: Callable

    @nn.compact
    def __call__(self, key, u, v, walk_receivers):
        # intentionally using the same key to get the same permutation here
        u_perm = jax.random.permutation(key, u)
        v_perm = jax.random.permutation(key, v)

        fe = GenewiseNodePairClassifier(training=self.training)

        score, penalty = fe(u, v[walk_receivers])
        perm_score, _ = fe(u_perm, v_perm[walk_receivers])

        mi_bounds = self.objective(score, perm_score)

        metrics = {"tp": score > perm_score}

        return mi_bounds - penalty, metrics


# Copying an idiom from: https://github.com/deepmind/optax/issues/197#issuecomment-974505149
def make_train_step(optimizer):
    @partial(jax.jit, static_argnums=(0,))
    def train_step(modelargs, u, v, walk_receivers, model_state, params, opt_state, key):
        def loss_fn(params):
            vars = {"params": params, **model_state}
            (mi_lower_bounds, metrics), new_model_state = MINE(
                training=True, **modelargs).apply(
                    vars, key, u, v, walk_receivers, mutable=["batch_stats"])
            return -jnp.mean(mi_lower_bounds), (mi_lower_bounds, metrics, new_model_state)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (neg_mean_mi_lower_bounds, (mi_lower_bounds, metrics, model_state)), grads = grad_fn(params)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return model_state, params, opt_state, mi_lower_bounds, metrics

    return train_step


@partial(jax.jit, static_argnums=(0,))
def eval_step(modelargs, vars, key, v, u, walk_receivers):
    (mi_lower_bounds, metrics), new_model_state = MINE(training=False, **modelargs).apply(
        vars, key,
        v, u,
        walk_receivers,
        mutable=["batch_stats"])

    return mi_lower_bounds, metrics["tp"]


@jax.jit
def sample_signals_gamma(key, v, v_mean, v_std, post_θ, prior_k):
    v_sample = jnp.log1p(post_θ * jax.random.gamma(key, v + prior_k))
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