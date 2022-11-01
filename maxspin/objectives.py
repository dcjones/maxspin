
import jax.numpy as jnp
import jax.nn as nn

def genewise_js(fa, fb):
    return jnp.sum(
        (-nn.softplus(-fa) - nn.softplus(fb) - jnp.log(0.25)) / (-jnp.log(0.25)),
        axis=0)

def genewise_dgi(fa, fb):
    return jnp.sum(nn.log_sigmoid(fa), axis=0) + fa.shape[0] * jnp.log(2) \
         + jnp.sum(nn.log_sigmoid(-fb), axis=0) + fb.shape[0] * jnp.log(2)