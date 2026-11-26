"""
Bit-exact integer reference model of the generated hardware.

Every stage works on integer tensors of shape ``(n, H, W, C)`` (stream shape
per sample) and is the specification its VHDL must match:

  linear       acc[j] = (Σ_i W[j,i]·x[i]) · 2^F + bias[j]      per beat
               acc = max(acc, 0) if relu;  out = requant(acc) or acc
  relu         max(0, x)
  pad          border of constant ``value``
  swg          one window beat per output position, elements in (kh, kw, c) order
  pool         per-channel max / sum over a window beat
  global_pool  per-channel sum over all beats
  flatten      (H, W, C) → (1, 1, H·W·C), HWC order
  lut          out = table[table_of[c]][x - in_lo]
  requant      out = clip(shift_round(x·m[c], n[c], R), lo, hi)

``generate_test_vectors`` produces stimulus for the VHDL testbench, and
``digitalify.verify`` compares this model against the QONNX execution of the
original model, closing the loop  QONNX ⇔ reference ⇔ VHDL.
"""

from __future__ import annotations

import numpy as np

from digitalify.ir.graph import IRGraph
from digitalify.ir.node import IRNode
from digitalify.quant import shift_round


def _requant(acc: np.ndarray, rq: dict) -> np.ndarray:
    """Per-channel requantisation of an ``(..., C)`` integer array."""
    mode = rq["rounding"]
    fn = np.frompyfunc(lambda a, m, n: shift_round(int(a) * int(m), int(n), mode), 3, 1)
    q = fn(acc, rq["multiplier"], rq["shift"]).astype(np.int64)
    return np.clip(q, rq["clip_min"], rq["clip_max"])


def _linear(x: np.ndarray, node: IRNode) -> np.ndarray:
    a = node.attrs
    acc = ((x @ a["weight"].T) << a.get("bias_frac_bits", 0)) + a["bias"]
    if a.get("relu"):
        acc = np.maximum(acc, 0)
    rq = a.get("requant")
    return acc if rq is None else _requant(acc, rq)


def _pad(x: np.ndarray, node: IRNode) -> np.ndarray:
    t, l, b, r = node.attrs["pads"]
    return np.pad(x, ((0, 0), (t, b), (l, r), (0, 0)), constant_values=node.attrs["value"])


def _swg(x: np.ndarray, node: IRNode) -> np.ndarray:
    kh, kw = node.attrs["kernel"]
    sh, sw = node.attrs["stride"]
    ho, wo, _ = node.output_shape
    out = np.empty((x.shape[0], ho, wo, kh * kw * x.shape[3]), dtype=np.int64)
    for oy in range(ho):
        for ox in range(wo):
            win = x[:, oy * sh: oy * sh + kh, ox * sw: ox * sw + kw, :]   # (n, kh, kw, C)
            out[:, oy, ox, :] = win.reshape(x.shape[0], -1)
    return out


def _pool(x: np.ndarray, node: IRNode) -> np.ndarray:
    c = node.out_features
    win = x.reshape(*x.shape[:3], -1, c)               # (n, H, W, KH·KW, C)
    return win.max(axis=3) if node.attrs["mode"] == "max" else win.sum(axis=3)


def _lut(x: np.ndarray, node: IRNode) -> np.ndarray:
    a = node.attrs
    idx = x - a["in_lo"]
    tables = np.asarray(a["tables"])
    table_of = np.asarray(a["table_of"])
    return tables[table_of[None, None, None, :], idx]


def _requant_node(x: np.ndarray, node: IRNode) -> np.ndarray:
    rq = node.attrs
    if rq.get("relu"):
        x = np.maximum(x, 0)
    return _requant(x, rq)


_STAGES = {
    "linear": _linear,
    "relu": lambda x, node: np.maximum(x, 0),
    "pad": _pad,
    "swg": _swg,
    "pool": _pool,
    "global_pool": lambda x, node: x.sum(axis=(1, 2), keepdims=True),
    "flatten": lambda x, node: x.reshape(x.shape[0], 1, 1, -1),
    "lut": _lut,
    "requant": _requant_node,
}


def run_reference(ir: IRGraph, inputs: np.ndarray) -> np.ndarray:
    """
    Run the integer reference model.

    Parameters
    ----------
    ir : IRGraph
    inputs : np.ndarray
        Integer inputs in stream order, shape ``(n_vectors, H·W·C)`` (or any
        shape with ``n_vectors`` first and ``H·W·C`` elements per sample).

    Returns
    -------
    np.ndarray
        int64 outputs in stream order, shape ``(n_vectors, H·W·C)``.
    """
    x = np.asarray(inputs, dtype=np.int64).reshape(len(inputs), *ir.input_stream)
    values = {"__input__": x}
    for node in ir.nodes:
        fn = _STAGES.get(node.op_type)
        if fn is None:
            raise ValueError(
                f"No reference implementation for op_type '{node.op_type}'. "
                "Add one in digitalify/testing/reference.py."
            )
        y = fn(values[node.input_names[0]], node)
        values[node.name] = np.asarray(y, dtype=np.int64).reshape(len(x), *node.output_shape)
    return values[ir.nodes[-1].name].reshape(len(x), -1)


def generate_test_vectors(
    ir: IRGraph,
    n_vectors: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integer input tensors and the reference outputs, in stream order.

    The first vectors are edge cases (all zeros, all max, all min); the rest
    are uniformly random over the input range.

    Returns
    -------
    (inputs, outputs) : int64 arrays of shape ``(n, in_size)`` and
    ``(n, out_size)``.
    """
    lo, hi = ir.input_range
    n_in = ir.in_size
    rng = np.random.default_rng(seed)
    edge = np.array([
        np.full(n_in, min(max(0, lo), hi)),
        np.full(n_in, hi),
        np.full(n_in, lo),
    ], dtype=np.int64)
    rand = rng.integers(lo, hi + 1, size=(max(0, n_vectors - len(edge)), n_in), dtype=np.int64)
    inputs = np.concatenate([edge, rand])[:n_vectors]
    return inputs, run_reference(ir, inputs)
