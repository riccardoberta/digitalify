"""
Python integer arithmetic reference model.

Replicates the exact fixed-point computation of the generated VHDL so that
expected outputs for test vectors can be computed without running a simulator.

The arithmetic mirrors the VHDL component by component:

  linear / quantized_linear
  ─────────────────────────
  acc[j] = bias[j] + Σ_i (weight[j,i] * x[i])      -- 32-bit accumulator
  out[j] = clamp(arithmetic_right_shift(acc[j], shift), -128, 127)

  relu
  ────
  out[i] = max(0, x[i])                              -- element-wise

Note on overflow: the VHDL accumulator is 32-bit signed.  Here we use 64-bit
integers so that the Python model never wraps; for networks with more than
~16 M INT8 inputs this could diverge from the VHDL, but that is beyond the
scope of typical test layers.
"""

from __future__ import annotations

import numpy as np

from digitalify.ir.graph import IRGraph
from digitalify.ir.node import IRNode


# ---------------------------------------------------------------------------
# Per-operator reference functions
# ---------------------------------------------------------------------------

def _linear_ref(x: np.ndarray, node: IRNode, data_width: int = 8) -> np.ndarray:
    """
    Reference for ``linear`` / ``quantized_linear``.

    Matches the VHDL:
      * combinational 64-bit MAC (64-bit avoids Python-side overflow)
      * arithmetic right-shift by ``shift`` bits
      * clamp to [-2^(data_width-1), 2^(data_width-1)-1]
    """
    weight: np.ndarray = node.attrs["weight"]   # (out_features, in_features) int8
    bias:   np.ndarray | None = node.attrs.get("bias")
    shift:  int = node.attrs.get("shift", 0)

    # Promote to int64 before multiplication to avoid overflow
    acc = weight.astype(np.int64) @ x.astype(np.int64)
    if bias is not None:
        acc = acc + bias.astype(np.int64)

    # Arithmetic right-shift: matches VHDL shift_right() on signed
    if shift > 0:
        acc = np.right_shift(acc, shift)

    # Saturate to data_width-bit signed range (matches VHDL clamp)
    lo = -(1 << (data_width - 1))
    hi =  (1 << (data_width - 1)) - 1
    return np.clip(acc, lo, hi).astype(np.int8)


def _relu_ref(x: np.ndarray) -> np.ndarray:
    """Reference for ``relu``: element-wise max(0, x), fully combinational."""
    return np.maximum(0, x).astype(np.int8)


# ---------------------------------------------------------------------------
# Graph-level runner
# ---------------------------------------------------------------------------

def run_reference(
    ir: IRGraph,
    inputs: np.ndarray,
    data_width: int = 8,
) -> np.ndarray:
    """
    Run the reference model on a batch of input vectors.

    Parameters
    ----------
    ir : IRGraph
        The computation graph describing the network.
    inputs : np.ndarray
        Shape ``(n_vectors, in_features)``, dtype ``int8``.
    data_width : int
        Bit-width of activations (default 8).

    Returns
    -------
    np.ndarray
        Shape ``(n_vectors, out_features)``, dtype ``int8``.

    Raises
    ------
    ValueError
        If the graph contains an operator without a reference implementation.
    """
    results = []
    for x in inputs:
        x = x.astype(np.int8)
        for node in ir.nodes:
            if node.op_type in ("linear", "quantized_linear"):
                x = _linear_ref(x, node, data_width=data_width)
            elif node.op_type == "relu":
                x = _relu_ref(x)
            else:
                raise ValueError(
                    f"No reference implementation for op_type '{node.op_type}'. "
                    "Add one in digitalify/testing/reference.py."
                )
        results.append(x)
    return np.array(results, dtype=np.int8)


# ---------------------------------------------------------------------------
# Test vector factory
# ---------------------------------------------------------------------------

def generate_test_vectors(
    ir: IRGraph,
    n_vectors: int = 8,
    data_width: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random INT8 input vectors and their expected outputs.

    The first three vectors are always edge-case inputs:
      - all zeros
      - all max (2^(data_width-1) - 1 = 127 for INT8)
      - all min (-128 for INT8)
    The remainder are random.

    Parameters
    ----------
    ir : IRGraph
        Used to determine ``input_shape`` and to run the reference model.
    n_vectors : int
        Total number of test vectors (at least 3 recommended).
    data_width : int
        Activation bit-width.
    seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    (inputs, outputs) : tuple[np.ndarray, np.ndarray]
        Both have shape ``(n_vectors, features)`` and dtype ``int8``.
    """
    in_features = ir.input_shape[-1]
    lo = -(1 << (data_width - 1))        # -128
    hi =  (1 << (data_width - 1)) - 1   #  127

    rng = np.random.default_rng(seed)

    # Edge cases
    edge = np.array([
        np.zeros(in_features, dtype=np.int8),               # all zeros
        np.full(in_features, hi,  dtype=np.int8),           # all max
        np.full(in_features, lo,  dtype=np.int8),           # all min
    ])

    n_random = max(0, n_vectors - len(edge))
    random_vecs = rng.integers(lo, hi + 1, size=(n_random, in_features), dtype=np.int8)

    inputs = np.concatenate([edge[:n_vectors], random_vecs], axis=0)[:n_vectors]
    outputs = run_reference(ir, inputs, data_width=data_width)
    return inputs, outputs
