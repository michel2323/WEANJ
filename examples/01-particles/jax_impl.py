"""JAX implementation of example 01 using jax-md's neighbour list (the idiomatic route).

The neighbour list is a fixed-capacity padded pair list: `allocate` (outside jit)
sizes it from the data, `update` (inside jit) refills it, and `did_buffer_overflow`
must be checked after each call, re-allocating with a larger capacity if needed.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

NUMBER_DENSITY = 10.0
H = 0.5
RC = 2 * H
MASS = 1.0
RHO0 = NUMBER_DENSITY


def box_size(n):
    return math.sqrt(n / NUMBER_DENSITY)


def W(r, h=H):
    """2D cubic spline (Monaghan 1992), support q = r/h in [0, 2]."""
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    t = jnp.maximum(2.0 - q, 0.0)
    return sigma * jnp.where(q < 1.0, 1.0 - 1.5 * q**2 + 0.75 * q**3, 0.25 * t**3)


def make_neighbor_fn(n, capacity_multiplier=1.25):
    L = box_size(n)
    displacement, _ = space.periodic(L)
    neighbor_fn = partition.neighbor_list(
        displacement, L, RC, dr_threshold=0.0,
        capacity_multiplier=capacity_multiplier, format=partition.Sparse)
    return displacement, neighbor_fn


def make_loss(n, displacement):
    """loss(x, nbrs) for positions x of shape (N, 2)."""
    w0 = MASS * float(W(0.0))  # self-contribution, not in jax-md's pair list

    def loss(x, nbrs):
        nbrs = nbrs.update(x)  # rebuild the cell list from the current positions
        i, j = nbrs.idx
        mask = i < n  # padded slots hold index n
        dR = jax.vmap(displacement)(x[i], x[j])
        # Padded slots give dR = 0; replace before the norm so its gradient is not NaN.
        dR = jnp.where(mask[:, None], dR, 1.0)
        r = jnp.sqrt(jnp.sum(dR**2, axis=1))
        w = jnp.where(mask, MASS * W(r), 0.0)
        rho = w0 + jax.ops.segment_sum(w, i, num_segments=n)
        return jnp.sum((rho - RHO0) ** 2), nbrs

    return loss


class Particles:
    """Bundle of jitted loss and gradient with jax-md's overflow handling."""

    def __init__(self, x, capacity_multiplier=1.25):
        self.n = x.shape[0]
        displacement, self.neighbor_fn = make_neighbor_fn(self.n, capacity_multiplier)
        self.nbrs = self.neighbor_fn.allocate(x)
        loss = make_loss(self.n, displacement)
        self._loss = jax.jit(loss)
        self._grad = jax.jit(jax.grad(loss, has_aux=True))

    def _checked(self, f, x):
        out, nbrs = f(x, self.nbrs)
        if nbrs.did_buffer_overflow:  # host sync; jax-md's documented pattern
            self.nbrs = self.neighbor_fn.allocate(x)
            out, nbrs = f(x, self.nbrs)
        self.nbrs = nbrs
        return out

    def loss(self, x):
        return self._checked(self._loss, x)

    def grad(self, x):
        return self._checked(self._grad, x)

    @property
    def capacity(self):
        return self.nbrs.idx.shape[1]


def positions(n, dtype=np.float32, seed=1):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random((n, 2)) * box_size(n), dtype=dtype)
