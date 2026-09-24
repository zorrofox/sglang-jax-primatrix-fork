"""silu_mul_rows matches the whole-buffer activation on the local rows and zeros the rest."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsv4.moe_act import silu_mul_rows
from sgl_jax.srt.layers.activation import silu_and_mul_with_clamp


def _tpu_generation() -> str:
    """ "v6e" / "v7x" / "other" from the local device kind ("TPU v6 lite", "TPU7x", ...)."""
    kind = jax.devices()[0].device_kind.lower()
    if "v6" in kind:
        return "v6e"
    if "7x" in kind or "v7" in kind:
        return "v7x"
    return "other"


@pytest.mark.parametrize(
    "rows,start,end", [(4096, 700, 2900), (4096, 0, 4096), (4096, 1024, 1024), (4000, 3500, 3999)]
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_rows_match(rows, start, end, dtype):
    k1, k2 = jax.random.split(jax.random.PRNGKey(rows + start))
    gate = (jax.random.normal(k1, (rows, 256), jnp.float32) * 8).astype(dtype)
    up = (jax.random.normal(k2, (rows, 256), jnp.float32) * 8).astype(dtype)
    ref = np.asarray(silu_and_mul_with_clamp(gate, up, 10.0), np.float32)
    out = np.asarray(
        jax.jit(lambda g, u, s, e: silu_mul_rows(g, u, s, e, limit=10.0, interpret=True))(
            gate, up, jnp.int32(start), jnp.int32(end)
        ),
        np.float32,
    )
    assert out.shape == ref.shape
    if dtype == jnp.bfloat16 and _tpu_generation() == "v6e":
        # On v6e the kernel and the XLA reference differ by one bf16 ulp (measured:
        # max abs 0.5 at |v| in [64, 128), max rel 2**-7), i.e. the silu is rounded
        # at different points on this generation. v7x and f32 stay bit-equal.
        np.testing.assert_allclose(out[start:end], ref[start:end], rtol=2**-7, atol=2**-6)
    else:
        np.testing.assert_array_equal(out[start:end], ref[start:end])
    block = 512
    first, last = start // block, (end + block - 1) // block - 1
    for b in range(-(-rows // block)):
        if b < first or b > last:
            assert not out[b * block : (b + 1) * block].any()
