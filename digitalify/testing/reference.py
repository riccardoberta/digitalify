"""
Bit-exact integer reference model of the generated hardware.

  linear
  ──────
  acc[j] = bias[j] + Σ_i weight[j,i] · x[i]               (exact, int64)
  acc[j] = max(acc[j], 0)                                  if relu
  out[j] = clip(shift_round(acc[j] · m[j], n[j], R), lo, hi)   if requant
  out[j] = acc[j]                                          otherwise

  relu
  ────
  out[i] = max(0, x[i])

``generate_test_vectors`` produces stimulus for the VHDL testbench, and
``digitalify.verify`` compares this model against the QONNX execution of the
original model, closing the loop  QONNX ⇔ reference ⇔ VHDL.
"""

from __future__ import annotations

import numpy as np

from digitalify.ir.graph import IRGraph
from digitalify.ir.node import IRNode
from digitalify.quant import shift_round


def _linear_ref(x: np.ndarray, node: IRNode) -> np.ndarray:
    w = node.attrs["weight"]
    acc = ((x @ w.T) << node.attrs.get("bias_frac_bits", 0)) + node.attrs["bias"]   # (n, out)
    if node.attrs.get("relu"):
        acc = np.maximum(acc, 0)
    rq = node.attrs.get("requant")
    if rq is None:
        return acc
    mode = rq["rounding"]
    fn = np.frompyfunc(lambda a, m, n: shift_round(int(a) * int(m), int(n), mode), 3, 1)
    q = fn(acc, rq["multiplier"][None, :], rq["shift"][None, :]).astype(np.int64)
    return np.clip(q, rq["clip_min"], rq["clip_max"])


def run_reference(ir: IRGraph, inputs: np.ndarray) -> np.ndarray:
    """
    Run the integer reference model.

    Parameters
    ----------
    ir : IRGraph
    inputs : np.ndarray
        Integer inputs, shape ``(n_vectors, *ir.input_shape)`` or
        ``(n_vectors, in_features)``.

    Returns
    -------
    np.ndarray
        int64 outputs, shape ``(n_vectors, out_features)``.
    """
    x = np.asarray(inputs, dtype=np.int64).reshape(len(inputs), -1)
    values = {"__input__": x}
    for node in ir.nodes:
        xin = values[node.input_names[0]]
        if node.op_type == "linear":
            y = _linear_ref(xin, node)
        elif node.op_type == "relu":
            y = np.maximum(xin, 0)
        else:
            raise ValueError(
                f"No reference implementation for op_type '{node.op_type}'. "
                "Add one in digitalify/testing/reference.py."
            )
        values[node.name] = y
    return values[ir.nodes[-1].name]


def generate_test_vectors(
    ir: IRGraph,
    n_vectors: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integer input vectors and the reference outputs.

    The first vectors are edge cases (all zeros, all max, all min); the rest
    are uniformly random over the input range.

    Returns
    -------
    (inputs, outputs) : int64 arrays of shape ``(n, in_features)`` and
    ``(n, out_features)``.
    """
    lo, hi = ir.input_range
    n_in = ir.in_features
    rng = np.random.default_rng(seed)
    edge = np.array([
        np.full(n_in, min(max(0, lo), hi)),
        np.full(n_in, hi),
        np.full(n_in, lo),
    ], dtype=np.int64)
    rand = rng.integers(lo, hi + 1, size=(max(0, n_vectors - len(edge)), n_in), dtype=np.int64)
    inputs = np.concatenate([edge, rand])[:n_vectors]
    return inputs, run_reference(ir, inputs)
