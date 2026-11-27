"""
QONNX → IRGraph importer.

Accepts a QONNX model (file path, ``onnx.ModelProto`` or qonnx
``ModelWrapper``) as exported by Brevitas (``export_qonnx``), converted from
QKeras, or built by hand, and lowers it to a chain of streaming hardware
stages (see ``docs/dataflow_design.md``).

The model is first normalised with ``qonnx.util.cleanup.cleanup_model``
(shape inference, constant folding with Quant nodes preserved, unique names,
batch size 1).  The graph is then walked from the single input to the single
output:

    input ─► Quant                                  network input quantiser
    MatMul | Gemm (Quant(W))                        ┐
    Conv (Quant(W)) → [pad] → [swg] →               ┘ → linear, fused with
        [Add | Sub | Mul | Div by constants | BatchNormalization]...
        [Relu] [Quant]
    MaxPool        → [pad] → swg → pool(max)
    AveragePool    →         swg → pool(sum)        (scale / window)
    GlobalAveragePool → global_pool                 (scale / H·W)
    Pad (constant 0)  → pad
    Flatten / Reshape of an image → flatten (+ weight column permutation)
    Relu                                            → relu
    [Relu] Quant                                    → requant
    any elementwise chain ending in Quant           → lut (table computed by
                                                      executing the chain)

Tensors are streams: an image ``(C, H, W)`` (or a sequence ``(C, L)``) is
streamed pixel by pixel as ``(H, W, C)``; anything else is one vector beat.

Integer semantics of a linear stage
-----------------------------------
With per-tensor input scale ``s_x``, per-output-channel weight scale
``s_w[j]``, folded affine factor ``a[j]`` / offset ``b[j]`` (bias, BatchNorm,
Mul / Add by constants) and output scale ``s_y[j]``:

    acc[j]   = Σ_i q_w[j,i]·q_x[i] · 2^F + round(b[j] / (|a[j]|·s_x·s_w[j]) · 2^F)
    q_y[j]   = clip(R(acc[j] · |a[j]|·s_x·s_w[j] / (2^F · s_y[j])))

(negative ``a[j]`` flips the sign of row j of the weights).  Without a
``Quant`` the accumulator itself is the output.

Limitations (``UnsupportedOperatorError`` / ``QONNXImportError``): single
input / output, no branching, zero-points 0, per-tensor activation scales
where activations are mixed (MatMul, Conv), no dilation, 1-bit bipolar
quantisation not mapped.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, replace
from fractions import Fraction
from math import prod

import numpy as np
import onnx
from onnx import helper

from digitalify.backend.vhdl.naming import vhdl_identifier
from digitalify.exceptions import QONNXImportError, UnsupportedOperatorError
from digitalify.ir.graph import IRGraph
from digitalify.ir.node import IRNode
from digitalify.ir.types import IntType
from digitalify.quant import normalize_rounding, quant_thresholds, requant_multiplier, to_fraction

# Ops that only change the logical shape of a tensor.
_PASSTHROUGH_OPS = {"Flatten", "Reshape", "Identity", "Squeeze", "Unsqueeze"}

# Ops that consume an image stream.
_SPATIAL_OPS = {"Conv", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "Pad"}

# Unary elementwise ops (optionally with constant operands) a LUT can absorb.
_ELEMENTWISE_OPS = {
    "Relu", "Sigmoid", "Tanh", "Clip", "LeakyRelu", "HardSigmoid", "HardSwish", "Elu", "Selu",
    "Softplus", "Softsign", "Abs", "Neg", "Exp", "Erf", "Gelu", "Mul", "Add", "Sub", "Div",
    "BatchNormalization", "Identity", "Quant",
}

# Bound on accumulator width so the int64 reference model cannot overflow.
_MAX_ACC_BITS = 62

# Largest input range (values per channel) turned into a lookup table.
MAX_LUT_ENTRIES = 4096

# Threshold activations: with thresholds="auto", a requantised linear output
# of at most AUTO_THRESHOLD_BITS bits uses 2^bits - 1 comparisons whenever the
# multiplier form would be inexact; thresholds=True uses them up to
# MAX_THRESHOLD_BITS bits.
AUTO_THRESHOLD_BITS = 4
MAX_THRESHOLD_BITS = 8


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_qonnx(model):
    """Return a cleaned-up qonnx ``ModelWrapper`` (batch size 1).

    *model* may be a path, an ``onnx.ModelProto`` or a ``ModelWrapper``;
    the caller's object is never modified.
    """
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup_model

    if isinstance(model, ModelWrapper):
        mw = ModelWrapper(copy.deepcopy(model.model))
    elif isinstance(model, onnx.ModelProto):
        mw = ModelWrapper(copy.deepcopy(model))
    elif isinstance(model, (str, os.PathLike)):
        mw = ModelWrapper(str(model))
    else:
        raise TypeError(
            "Expected a path, onnx.ModelProto or qonnx ModelWrapper, "
            f"got {type(model).__name__}"
        )

    # Some exporters (torch.onnx dynamo, used by Brevitas >= 0.12) also list
    # the initializers as graph inputs; they are constants, not inputs.
    consts = {init.name for init in mw.graph.initializer}
    for inp in [i for i in mw.graph.input if i.name in consts]:
        mw.graph.input.remove(inp)

    if len(mw.graph.input) != 1 or len(mw.graph.output) != 1:
        raise QONNXImportError(
            f"Only single-input / single-output models are supported "
            f"(got {len(mw.graph.input)} inputs, {len(mw.graph.output)} outputs)."
        )

    in_shape = mw.get_tensor_shape(mw.graph.input[0].name)
    needs_batch_fix = not in_shape or in_shape[0] != 1
    return cleanup_model(mw, override_inpsize=1 if needs_batch_fix else None)


# ---------------------------------------------------------------------------
# Quant node helpers
# ---------------------------------------------------------------------------

@dataclass
class _QuantParams:
    scale: np.ndarray          # float32, broadcastable
    bits: int
    signed: bool
    narrow: bool
    rounding: str              # as written in the model (for qonnx rounding fn)
    node_name: str

    @property
    def qmin(self) -> int:
        from qonnx.custom_op.general.quant import min_int
        return int(min_int(self.signed, self.narrow, self.bits))

    @property
    def qmax(self) -> int:
        from qonnx.custom_op.general.quant import max_int
        return int(max_int(self.signed, self.narrow, self.bits))

    @property
    def dtype(self) -> IntType:
        return IntType(self.bits, self.signed)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Integer image of *values*, computed exactly like qonnx (float32)."""
        from qonnx.custom_op.general.quant import resolve_rounding_mode
        y = values.astype(np.float32) / self.scale.astype(np.float32)
        y = np.clip(y, self.qmin, self.qmax)
        return resolve_rounding_mode(self.rounding)(y).astype(np.int64)


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            v = helper.get_attribute_value(a)
            return v.decode() if isinstance(v, bytes) else v
    return default


def _quant_params(mw, node) -> _QuantParams:
    if node.op_type == "BipolarQuant":
        raise UnsupportedOperatorError("BipolarQuant", node.name,
                                       "1-bit bipolar quantisation is not mapped yet.")
    scale = mw.get_initializer(node.input[1])
    zeropt = mw.get_initializer(node.input[2])
    bits = mw.get_initializer(node.input[3])
    if scale is None or zeropt is None or bits is None:
        raise QONNXImportError(
            f"Quant node '{node.name}': scale, zero-point and bit-width must be constants."
        )
    if np.any(zeropt != 0):
        raise QONNXImportError(
            f"Quant node '{node.name}': only zero-point 0 is supported (got {zeropt})."
        )
    if np.size(bits) != 1 or float(bits) != int(bits):
        raise QONNXImportError(f"Quant node '{node.name}': bit-width must be a scalar integer.")
    if np.any(scale <= 0):
        raise QONNXImportError(f"Quant node '{node.name}': scales must be positive.")
    signed = bool(_attr(node, "signed", 1))
    q = _QuantParams(
        scale=np.asarray(scale, dtype=np.float32),
        bits=int(bits),
        signed=signed,
        narrow=bool(_attr(node, "narrow", 0)),
        rounding=str(_attr(node, "rounding_mode", "ROUND")),
        node_name=node.name,
    )
    if q.bits == 1 and q.signed:
        raise UnsupportedOperatorError("Quant", node.name,
                                       "1-bit signed (bipolar) quantisation is not mapped yet.")
    normalize_rounding(q.rounding)   # validates
    return q


def _constant(mw, tensor: str):
    """Return ``(float_values, QuantParams | None)`` for a constant operand.

    The operand is either a plain initializer or a ``Quant`` node applied to an
    initializer (how Brevitas exports quantised weights).  Returns
    ``(None, None)`` if *tensor* is not constant.
    """
    init = mw.get_initializer(tensor)
    if init is not None:
        return np.asarray(init), None
    prod_node = mw.find_producer(tensor)
    if prod_node is not None and prod_node.op_type in ("Quant", "BipolarQuant"):
        src = mw.get_initializer(prod_node.input[0])
        if src is not None:
            return np.asarray(src), _quant_params(mw, prod_node)
    return None, None


def _integer_operand(values: np.ndarray, qp: _QuantParams | None, what: str, node_name: str):
    """Integer values and per-element scale of a constant operand."""
    if qp is not None:
        q = qp.quantize(values)
        scale = np.broadcast_to(qp.scale, values.shape).astype(np.float32)
        return q, scale
    if not np.all(np.mod(values, 1) == 0):
        raise QONNXImportError(
            f"{what} of node '{node_name}' are not quantised: feed them through a "
            "Quant node (or use integer-valued initializers)."
        )
    return values.astype(np.int64), np.ones(values.shape, dtype=np.float32)


def _dequantized(values: np.ndarray, qp: _QuantParams | None) -> np.ndarray:
    """Real value of a (possibly Quant-wrapped) constant, as qonnx computes it."""
    if qp is None:
        return values.astype(np.float32)
    return (qp.quantize(values).astype(np.float32) * qp.scale).astype(np.float32)


def _per_channel(values, n: int, what: str) -> np.ndarray:
    """Flatten a scalar or per-channel constant to a length-*n* vector."""
    v = np.asarray(values).reshape(-1)
    if v.size == 1:
        return np.full(n, v[0], dtype=v.dtype)
    if v.size != n:
        raise QONNXImportError(f"{what}: expected 1 or {n} values, got shape {np.shape(values)}.")
    return v


# ---------------------------------------------------------------------------
# Graph walking state
# ---------------------------------------------------------------------------

@dataclass
class _Tensor:
    """A quantised tensor flowing on the hardware stream."""
    name: str                       # ONNX tensor name
    onnx_shape: tuple[int, ...]     # batch dimension excluded
    stream: tuple[int, int, int]    # (H, W, C)
    spatial: bool                   # ONNX (C, *S) streamed as (H, W, C)
    dtype: IntType
    lo: int                         # tight integer range of the values
    hi: int
    scale: np.ndarray               # float64, one entry per stream channel
    src: str                        # producing IR node name, or "__input__"
    perm: np.ndarray | None = None  # flat tensors: stream element i = ONNX element perm[i]

    @property
    def channels(self) -> int:
        return int(self.stream[2])

    @property
    def pixels(self) -> int:
        return int(self.stream[0] * self.stream[1])

    def uniform_scale(self, what: str) -> float:
        if np.any(self.scale != self.scale[0]):
            raise QONNXImportError(
                f"{what}: the input has per-channel scales; this layer mixes channels and "
                "needs a single per-tensor scale (add a Quant node before it)."
            )
        return float(self.scale[0])

    def onnx_channel_scale(self) -> np.ndarray:
        """Scale per ONNX channel (dimension 1) of the tensor."""
        if self.perm is None:
            return self.scale
        out = np.empty_like(self.scale)
        out[self.perm] = self.scale
        return out


def _strip_batch(shape) -> tuple[int, ...]:
    shape = tuple(int(d) for d in (shape or ()))
    if len(shape) > 1 and shape[0] == 1:
        shape = shape[1:]
    return shape


def _hw(spatial_dims) -> tuple[int, int]:
    """(H, W) of the spatial dimensions of a (C, *S) tensor (1-D → H = 1)."""
    return (1, int(spatial_dims[0])) if len(spatial_dims) == 1 else (int(spatial_dims[0]), int(spatial_dims[1]))


def _single_consumer(mw, tensor: str):
    consumers = mw.find_consumers(tensor) or []
    if len(consumers) > 1:
        raise UnsupportedOperatorError(
            "branch", consumers[0].name,
            f"Tensor '{tensor}' feeds {len(consumers)} nodes; only chains are supported.",
        )
    return consumers[0] if consumers else None


def _ints(node, name, default):
    v = _attr(node, name, None)
    return [int(x) for x in v] if v is not None else list(default)


class _Builder:
    """Walks the ONNX graph and appends IR nodes."""

    def __init__(self, mw, ir: IRGraph, max_mult_bits: int, max_bias_frac_bits: int,
                 thresholds="auto"):
        if thresholds not in ("auto", True, False):
            raise ValueError(f"thresholds must be 'auto', True or False, got {thresholds!r}")
        self.thresholds = thresholds
        self.mw = mw
        self.ir = ir
        self.max_mult_bits = max_mult_bits
        self.max_bias_frac_bits = max_bias_frac_bits
        self.counters: dict[str, int] = {}

    # ---- helpers ------------------------------------------------------

    def new_name(self, op: str) -> str:
        idx = self.counters.get(op, 0)
        self.counters[op] = idx + 1
        return f"{op}_{idx}"

    def add(self, op: str, x: _Tensor, out_stream, out_dtype: IntType, attrs: dict,
            onnx_node: str | None = None) -> str:
        name = self.new_name(op)
        if onnx_node:
            attrs = {**attrs, "onnx_node": onnx_node}
        self.ir.add_node(IRNode(
            name=name, op_type=op, input_names=[x.src],
            input_shape=tuple(int(d) for d in x.stream), output_shape=tuple(int(d) for d in out_stream),
            input_dtype=x.dtype, output_dtype=out_dtype, attrs=attrs,
        ))
        return name

    def shape_of(self, tensor: str) -> tuple[int, ...]:
        return _strip_batch(self.mw.get_tensor_shape(tensor))

    # ---- spatial geometry ------------------------------------------------

    def _window(self, node, x: _Tensor, kernel, strides, pads, pad_value: int, onnx_node: str):
        """[pad] → [swg] for a kernel; returns the window tensor and (Ho, Wo)."""
        if not x.spatial:
            raise UnsupportedOperatorError(node.op_type, node.name, "Expected an image / sequence input.")
        nd = len(x.onnx_shape) - 1
        if len(kernel) != nd:
            raise UnsupportedOperatorError(node.op_type, node.name, f"Kernel {kernel} for {nd}-D data.")
        dil = _attr(node, "dilations", None)
        if dil is not None and any(int(d) != 1 for d in dil):
            raise UnsupportedOperatorError(node.op_type, node.name, "Dilation is not supported.")
        if str(_attr(node, "auto_pad", "NOTSET")) not in ("NOTSET", ""):
            raise UnsupportedOperatorError(node.op_type, node.name, "auto_pad is not supported; use pads.")
        if nd == 1:
            kernel, strides = [1, kernel[0]], [1, strides[0]]
            pads = [0, pads[0], 0, pads[1]]               # (top, left, bottom, right)
        else:
            pads = [pads[0], pads[1], pads[2], pads[3]]   # ONNX: begin_h, begin_w, end_h, end_w
        kh, kw = kernel
        sh, sw = strides
        t, l, b, r = pads
        h, w, c = x.stream

        if any(pads):
            hp, wp = h + t + b, w + l + r
            src = self.add("pad", x, (hp, wp, c), x.dtype,
                           {"pads": (t, l, b, r), "value": int(pad_value)}, onnx_node)
            x = replace(x, stream=(hp, wp, c), src=src,
                        lo=min(x.lo, pad_value), hi=max(x.hi, pad_value))
            h, w = hp, wp
        if kh > h or kw > w:
            raise QONNXImportError(f"Node '{node.name}': kernel {kernel} larger than the input {(h, w)}.")
        ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
        if (kh, kw, sh, sw) != (1, 1, 1, 1):
            src = self.add("swg", x, (ho, wo, kh * kw * c), x.dtype,
                           {"kernel": (kh, kw), "stride": (sh, sw)}, onnx_node)
            x = replace(x, stream=(ho, wo, kh * kw * c), src=src,
                        scale=np.tile(x.scale, kh * kw))
        return x, (ho, wo), (kh, kw)

    # ---- linear (MatMul / Gemm / Conv) ---------------------------------------

    def linear(self, node, x: _Tensor) -> _Tensor:
        mw = self.mw
        bias_f = None
        if node.op_type == "MatMul":
            if node.input[0] != x.name:
                raise UnsupportedOperatorError("MatMul", node.name, "Only activation @ weights is supported.")
            wf, wq = _constant(mw, node.input[1])
            if wf is None or wf.ndim != 2:
                raise UnsupportedOperatorError("MatMul", node.name, "Weights must be a constant 2-D matrix.")
            q_w, s_w = _integer_operand(wf, wq, "Weights", node.name)
            q_w, s_w = q_w.T, s_w.T                                   # → (out, in)
        else:  # Gemm
            if node.input[0] != x.name or _attr(node, "transA", 0):
                raise UnsupportedOperatorError("Gemm", node.name, "Only A=activation, transA=0 is supported.")
            if float(_attr(node, "alpha", 1.0)) != 1.0:
                raise UnsupportedOperatorError("Gemm", node.name, "alpha != 1 is not supported.")
            wf, wq = _constant(mw, node.input[1])
            if wf is None or wf.ndim != 2:
                raise UnsupportedOperatorError("Gemm", node.name, "B must be a constant 2-D matrix.")
            q_w, s_w = _integer_operand(wf, wq, "Weights", node.name)
            if not _attr(node, "transB", 0):
                q_w, s_w = q_w.T, s_w.T
            if len(node.input) > 2 and node.input[2]:
                cf, cq = _constant(mw, node.input[2])
                if cf is None:
                    raise UnsupportedOperatorError("Gemm", node.name, "C (bias) must be constant.")
                if float(_attr(node, "beta", 1.0)) != 1.0:
                    raise UnsupportedOperatorError("Gemm", node.name, "beta != 1 is not supported.")
                bias_f = _dequantized(cf, cq)

        if x.pixels != 1:
            raise UnsupportedOperatorError(node.op_type, node.name,
                                           "MatMul / Gemm on an image: flatten it first.")
        out_f, in_f = q_w.shape
        if in_f != x.channels:
            raise QONNXImportError(
                f"Node '{node.name}': weight matrix expects {in_f} inputs, the stream carries {x.channels}."
            )
        if x.perm is not None:                     # hardware flattens in HWC order
            q_w, s_w = q_w[:, x.perm], s_w[:, x.perm]
        s_w_out = s_w[:, 0]
        if np.any(s_w != s_w_out[:, None]):
            raise QONNXImportError(
                f"Node '{node.name}': weight scales must be per-tensor or per output channel."
            )
        return self._linear_tail(node, x, q_w, s_w_out, bias_f, (1, 1), node.output[0])

    def conv(self, node, x: _Tensor) -> _Tensor:
        mw = self.mw
        wf, wq = _constant(mw, node.input[1])
        if wf is None or wf.ndim not in (3, 4):
            raise UnsupportedOperatorError("Conv", node.name, "Weights must be a constant 3-D / 4-D tensor.")
        q_w, s_w = _integer_operand(wf, wq, "Weights", node.name)
        c_out, c_in_g = q_w.shape[:2]
        if q_w.ndim == 3:                                 # 1-D conv → kernel (1, k)
            q_w, s_w = q_w[:, :, None, :], s_w[:, :, None, :]
        kh, kw = q_w.shape[2:]
        groups = int(_attr(node, "group", 1))
        c_in = x.channels
        if c_in_g * groups != c_in or c_out % groups:
            raise QONNXImportError(f"Node '{node.name}': {groups} groups do not match the channels.")
        s_w_out = s_w.reshape(c_out, -1)[:, 0]
        if np.any(s_w.reshape(c_out, -1) != s_w_out[:, None]):
            raise QONNXImportError(f"Node '{node.name}': weight scales must be per output channel.")
        bias_f = None
        if len(node.input) > 2 and node.input[2]:
            bf, bq = _constant(mw, node.input[2])
            if bf is None:
                raise UnsupportedOperatorError("Conv", node.name, "The bias must be constant.")
            bias_f = _dequantized(bf, bq)

        nd = len(x.onnx_shape) - 1
        kernel = [int(k) for k in wf.shape[2:]]
        strides = _ints(node, "strides", [1] * nd)
        pads = _ints(node, "pads", [0] * (2 * nd))
        win, (ho, wo), _ = self._window(node, x, kernel, strides, pads, 0, node.name)

        # Weight matrix in window order (kh, kw, c); groups → block-sparse.
        w_full = np.zeros((c_out, kh, kw, c_in), dtype=np.int64)
        co_g = c_out // groups
        for o in range(c_out):
            g = o // co_g
            w_full[o, :, :, g * c_in_g:(g + 1) * c_in_g] = q_w[o].transpose(1, 2, 0)
        return self._linear_tail(node, win, w_full.reshape(c_out, -1), s_w_out, bias_f, (ho, wo),
                                 node.output[0])

    def _linear_tail(self, node, x: _Tensor, q_w, s_w_out, bias_f, out_hw, cur: str) -> _Tensor:
        """Fuse [Add|Sub|Mul|Div const | BatchNorm]... [Relu] [Quant] after a MatMul / Conv."""
        mw = self.mw
        out_f = q_w.shape[0]
        s_x = to_fraction(x.uniform_scale(f"Node '{node.name}'"))

        # ---- affine chain: real = a · (s_x·s_w·Σwx) + b ----------------------
        a = [Fraction(1)] * out_f
        b = [Fraction(0)] * out_f
        if bias_f is not None:
            bv = _per_channel(bias_f, out_f, f"Bias of node '{node.name}'")
            b = [Fraction(float(v)) for v in bv]
        nxt = _single_consumer(mw, cur)
        while nxt is not None:
            op = nxt.op_type
            if op in ("Add", "Sub", "Mul", "Div") and cur in nxt.input[:2]:
                other = nxt.input[1] if nxt.input[0] == cur else nxt.input[0]
                cf, cq = _constant(mw, other)
                if cf is None or (op in ("Sub", "Div") and nxt.input[0] != cur):
                    break
                c = [Fraction(float(v)) for v in _per_channel(_dequantized(cf, cq), out_f,
                                                              f"Constant of node '{nxt.name}'")]
                if op == "Add":
                    b = [bj + cj for bj, cj in zip(b, c)]
                elif op == "Sub":
                    b = [bj - cj for bj, cj in zip(b, c)]
                elif op == "Mul":
                    a = [aj * cj for aj, cj in zip(a, c)]
                    b = [bj * cj for bj, cj in zip(b, c)]
                else:
                    if any(cj == 0 for cj in c):
                        raise QONNXImportError(f"Node '{nxt.name}': division by zero.")
                    a = [aj / cj for aj, cj in zip(a, c)]
                    b = [bj / cj for bj, cj in zip(b, c)]
            elif op == "BatchNormalization" and nxt.input[0] == cur:
                vals = [_constant(mw, t)[0] for t in nxt.input[1:5]]
                if any(v is None for v in vals):
                    raise UnsupportedOperatorError(op, nxt.name, "BatchNorm parameters must be constant.")
                gamma, beta, mean, var = (np.asarray(v, dtype=np.float32).reshape(-1) for v in vals)
                eps = np.float32(_attr(nxt, "epsilon", 1e-5))
                k = (gamma / np.sqrt(var + eps)).astype(np.float32)
                k = _per_channel(k, out_f, f"BatchNorm '{nxt.name}'")
                off = _per_channel((beta - mean * k).astype(np.float32), out_f, f"BatchNorm '{nxt.name}'")
                a = [aj * Fraction(float(kj)) for aj, kj in zip(a, k)]
                b = [bj * Fraction(float(kj)) + Fraction(float(oj)) for bj, kj, oj in zip(b, k, off)]
            else:
                break
            cur, nxt = nxt.output[0], _single_consumer(mw, nxt.output[0])

        q_w = q_w.astype(np.int64).copy()
        acc_unit = []
        for j in range(out_f):
            if a[j] == 0:
                raise QONNXImportError(f"Node '{node.name}': channel {j} is scaled by zero.")
            if a[j] < 0:
                q_w[j] = -q_w[j]
            acc_unit.append(abs(a[j]) * s_x * to_fraction(s_w_out[j]))

        relu = False
        if nxt is not None and nxt.op_type == "Relu":
            relu = True
            cur, nxt = nxt.output[0], _single_consumer(mw, nxt.output[0])

        unit = list(acc_unit)        # real value of one Σ w·x step, per channel

        # ---- bias in accumulator units ---------------------------------------
        # A bias that is not an integer multiple of the accumulator scale gets F
        # fractional bits:  acc = (Σ w·x) · 2^F + round(b / unit · 2^F)
        bias = np.zeros(out_f, dtype=np.int64)
        frac_bits = 0
        bias_exact = True
        if any(bj != 0 for bj in b):
            v = [b[j] / acc_unit[j] for j in range(out_f)]
            while frac_bits < self.max_bias_frac_bits and any((vj * (1 << frac_bits)).denominator != 1 for vj in v):
                frac_bits += 1
            scaled = [vj * (1 << frac_bits) for vj in v]
            bias_exact = all(s.denominator == 1 for s in scaled)
            bias[:] = [round(s) for s in scaled]
        acc_unit = [u / (1 << frac_bits) for u in acc_unit]

        # ---- accumulator range ------------------------------------------------
        in_f = q_w.shape[1]
        w_dtype = IntType.for_range(min(int(q_w.min()), 0), max(int(q_w.max()), 0), signed=True)
        if x.dtype.bits + 1 + w_dtype.bits + int(in_f).bit_length() + frac_bits > _MAX_ACC_BITS:
            raise QONNXImportError(f"Node '{node.name}': accumulator would exceed {_MAX_ACC_BITS} bits.")
        t_lo, t_hi = q_w * x.lo, q_w * x.hi
        acc_hi = (np.maximum(t_lo, t_hi).sum(axis=1) << frac_bits) + bias
        acc_lo = (np.minimum(t_lo, t_hi).sum(axis=1) << frac_bits) + bias
        lo, hi = int(acc_lo.min()), int(acc_hi.max())
        # Partial sums stay inside [s_lo, s_hi] because every input range contains 0.
        s_lo = int(np.minimum(t_lo, t_hi).sum(axis=1).min())
        s_hi = int(np.maximum(t_lo, t_hi).sum(axis=1).max())
        sum_bits = max(IntType.for_range(s_lo, s_hi, signed=True).bits,
                       x.dtype.bits + 1 + w_dtype.bits)              # product never truncated
        acc_bits = max(IntType.for_range(lo, hi, signed=True).bits, sum_bits + frac_bits)
        if acc_bits > _MAX_ACC_BITS:
            raise QONNXImportError(f"Node '{node.name}': accumulator would exceed {_MAX_ACC_BITS} bits.")
        acc_dtype = IntType(acc_bits, True)
        if relu:
            lo, hi = max(lo, 0), max(hi, 0)

        # ---- optional output Quant (requantisation) -----------------------------
        requant = None
        if nxt is not None and nxt.op_type in ("Quant", "BipolarQuant") and nxt.input[0] == cur:
            qp = _quant_params(mw, nxt)
            s_y = _per_channel(qp.scale, out_f, f"Scale of Quant node '{nxt.name}'")
            requant = self._requant_params(qp, [acc_unit[j] / to_fraction(s_y[j]) for j in range(out_f)])
            out_dtype = qp.dtype
            lo = max(qp.qmin, 0) if relu else qp.qmin
            hi = qp.qmax
            out_scale = s_y.astype(np.float64)
            cur = nxt.output[0]
            if self._use_thresholds(qp, requant["exact"] and bias_exact):
                s_los = np.minimum(t_lo, t_hi).sum(axis=1)
                s_his = np.maximum(t_lo, t_hi).sum(axis=1)
                thr, base = self._thresholds(unit, b, [to_fraction(v) for v in s_y],
                                             requant["rounding"], lo, hi, s_los, s_his)
                t_min = min([int(s_los.min())] + [int(v) for v in thr.reshape(-1)])
                t_max = max([int(s_his.max())] + [int(v) for v in thr.reshape(-1)])
                acc_bits = max(IntType.for_range(t_min, t_max, signed=True).bits, sum_bits)
                src = self.add("linear", x, (*out_hw, out_f), out_dtype, {
                    "weight": q_w,
                    "bias": np.zeros(out_f, dtype=np.int64),
                    "bias_exact": True,
                    "bias_frac_bits": 0,
                    "weight_dtype": w_dtype,
                    "sum_dtype": IntType(sum_bits, True),
                    "acc_dtype": IntType(acc_bits, True),
                    "relu": False,
                    "requant": None,
                    "thresholds": thr,
                    "threshold_base": base,
                    "rounding": requant["rounding"],
                }, node.name)
                return _Tensor(cur, self.shape_of(cur), (*out_hw, out_f), x.spatial or out_hw != (1, 1),
                               out_dtype, lo, hi, out_scale, src)
        else:
            out_dtype = IntType.for_range(lo, hi)
            out_scale = np.array([float(u) for u in acc_unit])

        out_stream = (*out_hw, out_f)
        src = self.add("linear", x, out_stream, out_dtype, {
            "weight": q_w,
            "bias": bias,
            "bias_exact": bias_exact,
            "bias_frac_bits": frac_bits,
            "weight_dtype": w_dtype,
            "sum_dtype": IntType(sum_bits, True),
            "acc_dtype": acc_dtype,
            "relu": relu,
            "requant": requant,
        }, node.name)
        return _Tensor(cur, self.shape_of(cur), out_stream, x.spatial or out_hw != (1, 1),
                       out_dtype, lo, hi, out_scale, src)

    def _use_thresholds(self, qp: _QuantParams, exact: bool) -> bool:
        if self.thresholds is False or qp.bits > MAX_THRESHOLD_BITS:
            return False
        if self.thresholds is True:
            return True
        return not exact and qp.bits <= AUTO_THRESHOLD_BITS

    @staticmethod
    def _thresholds(unit, bias, s_y, rounding, out_lo, out_hi, s_los, s_his):
        """Per-channel thresholds on the raw sum S = Σ w·x.

        ``out = base[j] + #{k : S >= T[j][k]}``.  Thresholds that every
        reachable S passes are folded into ``base``; the rows are padded to
        the same length with ``s_hi + 1``, which no reachable S reaches.
        """
        rows, base = [], []
        for j in range(len(unit)):
            t = quant_thresholds(unit[j], bias[j], s_y[j], rounding, out_lo, out_hi)
            s_lo, s_hi = int(s_los[j]), int(s_his[j])
            base.append(out_lo + sum(1 for v in t if v <= s_lo))
            rows.append([v for v in t if s_lo < v <= s_hi])
        n = max(1, max(len(r) for r in rows))
        thr = np.array([r + [int(s_his[j]) + 1] * (n - len(r)) for j, r in enumerate(rows)],
                       dtype=np.int64).reshape(len(rows), n)
        return thr, np.array(base, dtype=np.int64)

    def _requant_params(self, qp: _QuantParams, ratios) -> dict:
        mults, shifts, exact = [], [], True
        for r in ratios:
            m, n, ex = requant_multiplier(r, self.max_mult_bits)
            mults.append(m)
            shifts.append(n)
            exact &= ex
        return {
            "multiplier": np.array(mults, dtype=np.int64),
            "shift": np.array(shifts, dtype=np.int64),
            "rounding": normalize_rounding(qp.rounding),
            "clip_min": qp.qmin,
            "clip_max": qp.qmax,
            "exact": exact,
        }

    # ---- pooling ----------------------------------------------------------------

    def pool(self, node, x: _Tensor) -> _Tensor:
        op = node.op_type
        nd = len(x.onnx_shape) - 1
        kernel = _ints(node, "kernel_shape", [1] * nd)
        strides = _ints(node, "strides", [1] * nd)
        pads = _ints(node, "pads", [0] * (2 * nd))
        if int(_attr(node, "ceil_mode", 0)):
            raise UnsupportedOperatorError(op, node.name, "ceil_mode=1 is not supported.")
        if op == "AveragePool" and any(pads):
            raise UnsupportedOperatorError(op, node.name, "AveragePool with padding is not supported.")
        if op == "MaxPool" and len(node.output) > 1 and node.output[1]:
            raise UnsupportedOperatorError(op, node.name, "The Indices output is not supported.")
        win, (ho, wo), (kh, kw) = self._window(node, x, kernel, strides, pads, x.lo, node.name)
        k = kh * kw
        c = x.channels
        if op == "MaxPool":
            dtype, lo, hi, scale = x.dtype, x.lo, x.hi, x.scale
            mode = "max"
        else:
            lo, hi = x.lo * k, x.hi * k
            dtype = IntType.for_range(lo, hi)
            scale = x.scale / k
            mode = "sum"
        if win.src == x.src:                      # 1x1 window, stride 1: identity
            return replace(x, name=node.output[0], onnx_shape=self.shape_of(node.output[0]),
                           lo=lo, hi=hi, scale=scale, dtype=dtype if mode == "max" else x.dtype)
        src = self.add("pool", win, (ho, wo, c), dtype, {"mode": mode, "window": k}, node.name)
        return _Tensor(node.output[0], self.shape_of(node.output[0]), (ho, wo, c), True,
                       dtype, lo, hi, scale, src)

    def global_pool(self, node, x: _Tensor) -> _Tensor:
        if not x.spatial:
            raise UnsupportedOperatorError(node.op_type, node.name, "Expected an image / sequence input.")
        n = x.pixels
        c = x.channels
        if node.op_type == "GlobalMaxPool":
            raise UnsupportedOperatorError(node.op_type, node.name, "GlobalMaxPool is not mapped yet.")
        lo, hi = x.lo * n, x.hi * n
        dtype = IntType.for_range(lo, hi)
        scale = x.scale / n
        src = self.add("global_pool", x, (1, 1, c), dtype, {"mode": "sum"}, node.name)
        return _Tensor(node.output[0], self.shape_of(node.output[0]), (1, 1, c), True,
                       dtype, lo, hi, scale, src)

    def pad(self, node, x: _Tensor) -> _Tensor:
        mode = str(_attr(node, "mode", "constant"))
        init = self.mw.get_initializer(node.input[1]) if len(node.input) > 1 and node.input[1] else None
        pads = [int(p) for p in (init if init is not None else _attr(node, "pads", []))]
        value = 0.0
        if len(node.input) > 2 and node.input[2]:
            v = self.mw.get_initializer(node.input[2])
            value = float(np.asarray(v).reshape(-1)[0]) if v is not None else None
        if mode != "constant" or value != 0.0:
            raise UnsupportedOperatorError("Pad", node.name, "Only constant zero padding is supported.")
        rank = len(pads) // 2
        begin, end = pads[:rank], pads[rank:]
        nd = len(x.onnx_shape) - 1
        if any(begin[:rank - nd]) or any(end[:rank - nd]):
            raise UnsupportedOperatorError("Pad", node.name, "Only spatial dimensions can be padded.")
        begin, end = begin[rank - nd:], end[rank - nd:]
        if nd == 1:
            t, l, b, r = 0, begin[0], 0, end[0]
        else:
            t, l, b, r = begin[0], begin[1], end[0], end[1]
        h, w, c = x.stream
        out = (h + t + b, w + l + r, c)
        src = self.add("pad", x, out, x.dtype, {"pads": (t, l, b, r), "value": 0}, node.name)
        return replace(x, name=node.output[0], onnx_shape=self.shape_of(node.output[0]), stream=out,
                       src=src, lo=min(x.lo, 0), hi=max(x.hi, 0))

    # ---- shape ops --------------------------------------------------------------

    def reshape(self, node, x: _Tensor) -> _Tensor:
        new_shape = self.shape_of(node.output[0])
        if int(prod(new_shape)) != int(prod(x.onnx_shape)):
            raise QONNXImportError(f"Node '{node.name}' ({node.op_type}) changes the number of elements.")
        if x.spatial and new_shape == x.onnx_shape:
            return replace(x, name=node.output[0])
        if x.spatial and x.pixels > 1:
            if len(new_shape) > 1 and new_shape[0] != x.onnx_shape[0]:
                raise UnsupportedOperatorError(node.op_type, node.name,
                                               "Only flattening an image to a vector is supported.")
            if len(new_shape) > 1:
                raise UnsupportedOperatorError(node.op_type, node.name,
                                               f"Reshape of an image to {new_shape} is not supported.")
            # Stream element (h, w, c) is ONNX element c·H·W + h·W + w.
            h, w, c = x.stream
            hh, ww, cc = np.meshgrid(np.arange(h), np.arange(w), np.arange(c), indexing="ij")
            perm = (cc * h * w + hh * w + ww).reshape(-1)
            src = self.add("flatten", x, (1, 1, h * w * c), x.dtype, {}, node.name)
            return _Tensor(node.output[0], new_shape, (1, 1, h * w * c), False, x.dtype, x.lo, x.hi,
                           np.tile(x.scale, h * w), src, perm)
        # A vector (or a 1x1 image) stays one beat.
        return replace(x, name=node.output[0], onnx_shape=new_shape, spatial=False)

    # ---- elementwise chains ----------------------------------------------------

    def elementwise(self, node, x: _Tensor) -> _Tensor:
        """A chain of elementwise ops: relu, requant or lut."""
        mw = self.mw
        chain = [node]
        cur = node.output[0]
        while chain[-1].op_type not in ("Quant", "BipolarQuant"):
            nxt = _single_consumer(mw, cur)
            if nxt is None or nxt.op_type not in _ELEMENTWISE_OPS or not self._unary(nxt, cur):
                break
            chain.append(nxt)
            cur = nxt.output[0]
        ops = [n.op_type for n in chain]

        if ops == ["Relu"]:
            src = self.add("relu", x, x.stream, x.dtype, {}, node.name)
            return replace(x, name=cur, onnx_shape=self.shape_of(cur), src=src,
                           lo=max(x.lo, 0), hi=max(x.hi, 0))
        if ops[-1] not in ("Quant", "BipolarQuant"):
            raise UnsupportedOperatorError(
                ops[-1], chain[-1].name,
                f"The elementwise chain {ops} must end with a Quant node to be mapped "
                "(it becomes a lookup table).")
        qp = _quant_params(mw, chain[-1])
        c = x.channels
        s_y = self._stream_channels(x, _per_channel(qp.scale, self._onnx_channels(x),
                                                    f"Scale of Quant node '{chain[-1].name}'"))
        if ops in (["Quant"], ["Relu", "Quant"]):
            relu = ops[0] == "Relu"
            ratios = [to_fraction(x.scale[i]) / to_fraction(s_y[i]) for i in range(c)]
            attrs = self._requant_params(qp, ratios)
            attrs["relu"] = relu
            src = self.add("requant", x, x.stream, qp.dtype, attrs, chain[-1].name)
            lo_out = max(qp.qmin, 0) if relu else qp.qmin
        else:
            tables, table_of = self._lut_tables(chain, x, qp)
            src = self.add("lut", x, x.stream, qp.dtype,
                           {"tables": tables, "table_of": table_of, "in_lo": x.lo}, chain[0].name)
            lo_out = int(tables[np.unique(table_of)].min())
        return replace(x, name=cur, onnx_shape=self.shape_of(cur), src=src, dtype=qp.dtype,
                       lo=lo_out, hi=qp.qmax, scale=s_y.astype(np.float64))

    def _unary(self, node, cur: str) -> bool:
        """True if *node* uses *cur* as its only non-constant input."""
        if node.op_type in ("Quant", "BipolarQuant"):
            return node.input[0] == cur
        for t in node.input:
            if t and t != cur and _constant(self.mw, t)[0] is None:
                return False
        return cur in node.input

    @staticmethod
    def _onnx_channels(x: _Tensor) -> int:
        """Size of the axis per-channel constants refer to: C of an image,
        every element of a vector."""
        return int(x.onnx_shape[0]) if x.spatial else int(prod(x.onnx_shape))

    @staticmethod
    def _stream_channels(x: _Tensor, per_onnx: np.ndarray) -> np.ndarray:
        """Per-ONNX-channel values → per-stream-channel values."""
        v = np.asarray(per_onnx)
        return v[x.perm] if x.perm is not None else v

    def _lut_tables(self, chain, x: _Tensor, qp: _QuantParams):
        """Run the chain with qonnx on every input value of every channel."""
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.core.onnx_exec import execute_onnx
        from qonnx.transformation.infer_shapes import InferShapes

        n_vals = x.hi - x.lo + 1
        if n_vals > MAX_LUT_ENTRIES:
            raise UnsupportedOperatorError(
                chain[0].op_type, chain[0].name,
                f"Input range of {n_vals} values is too large for a lookup table "
                f"(max {MAX_LUT_ENTRIES}); quantise the input of this activation to fewer bits.")
        c_onnx = self._onnx_channels(x)
        scale_onnx = x.scale if x.spatial else x.onnx_channel_scale()
        q = np.arange(x.lo, x.hi + 1, dtype=np.float64)
        vals = (q[:, None] * scale_onnx[None, :]).astype(np.float32)          # (values, C)
        if x.spatial:
            vals = vals.reshape(n_vals, c_onnx, *([1] * (len(x.onnx_shape) - 1)))
        else:
            vals = vals.reshape(n_vals, *x.onnx_shape)

        inits, names = [], set()
        for n in chain:
            for t in n.input:
                if t and t not in names and self.mw.get_initializer(t) is not None:
                    inits.append(onnx.numpy_helper.from_array(self.mw.get_initializer(t), t))
                    names.add(t)
                elif t and t not in names and t != chain[0].input[0]:
                    prod_node = self.mw.find_producer(t)      # Quant-wrapped constant
                    if prod_node is not None and prod_node not in chain:
                        raise UnsupportedOperatorError(n.op_type, n.name, "Non-constant operand in the chain.")
        out = chain[-1].output[0]
        graph = helper.make_graph(
            [copy.deepcopy(n) for n in chain], "lut_eval",
            [helper.make_tensor_value_info(chain[0].input[0], onnx.TensorProto.FLOAT, list(vals.shape))],
            [helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None)],
            initializer=inits,
        )
        model = helper.make_model(graph, opset_imports=list(self.mw.model.opset_import))
        model.ir_version = self.mw.model.ir_version
        sub = ModelWrapper(model).transform(InferShapes())
        y = np.asarray(execute_onnx(sub, {chain[0].input[0]: vals})[out], dtype=np.float64)
        y = y.reshape(n_vals, c_onnx)
        s_y_onnx = _per_channel(qp.scale, c_onnx, "Quant scale").astype(np.float64)
        table_onnx = np.rint(y / s_y_onnx[None, :]).astype(np.int64)          # (values, C)
        if x.perm is not None:
            table_onnx = table_onnx[:, x.perm]
        tables, table_of, seen = [], [], {}
        for ch in range(x.channels):
            key = table_onnx[:, ch].tobytes()
            if key not in seen:
                seen[key] = len(tables)
                tables.append(table_onnx[:, ch].copy())
            table_of.append(seen[key])
        return np.array(tables, dtype=np.int64), np.array(table_of, dtype=np.int64)


# ---------------------------------------------------------------------------
# Network input
# ---------------------------------------------------------------------------

def _network_input(mw) -> _Tensor:
    """Quantised view of the graph input."""
    in_name = mw.graph.input[0].name
    shape = _strip_batch(mw.get_tensor_shape(in_name))
    first = _single_consumer(mw, in_name)

    def make(name, dtype, lo, hi, scale):
        nxt = _single_consumer(mw, name)
        spatial = len(shape) in (2, 3) and nxt is not None and nxt.op_type in _SPATIAL_OPS
        if spatial:
            h, w = _hw(shape[1:])
            stream = (h, w, int(shape[0]))
        else:
            stream = (1, 1, int(prod(shape)))
        return _Tensor(name, shape, stream, spatial, dtype, lo, hi,
                       np.full(stream[2], scale, dtype=np.float64), "__input__")

    if first is not None and first.op_type in ("Quant", "BipolarQuant") and first.input[0] == in_name:
        qp = _quant_params(mw, first)
        if np.size(qp.scale) != 1:
            raise QONNXImportError(
                f"Input Quant '{first.name}': only a per-tensor input scale is supported."
            )
        return make(first.output[0], qp.dtype, qp.qmin, qp.qmax, float(qp.scale.reshape(-1)[0]))

    # FINN-style: input annotated with an integer DataType and no Quant node.
    dt = mw.get_tensor_datatype(in_name)
    if dt is not None and dt.is_integer() and dt.bitwidth() > 1:
        it = IntType(int(dt.bitwidth()), bool(dt.signed()))
        return make(in_name, it, int(dt.min()), int(dt.max()), 1.0)

    raise QONNXImportError(
        "The network input must be quantised: the first node should be a Quant node "
        "(e.g. Brevitas QuantIdentity) or the input must carry an integer QONNX DataType."
    )


def _input_rounding(mw) -> str:
    first = _single_consumer(mw, mw.graph.input[0].name)
    if first is not None and first.op_type == "Quant":
        return normalize_rounding(str(_attr(first, "rounding_mode", "ROUND")))
    return "ROUND"


# ---------------------------------------------------------------------------
# Main importer
# ---------------------------------------------------------------------------

def from_qonnx(
    model,
    name: str | None = None,
    max_mult_bits: int = 32,
    max_bias_frac_bits: int = 16,
    thresholds="auto",
) -> IRGraph:
    """
    Convert a QONNX model into an ``IRGraph``.

    Parameters
    ----------
    model : str | os.PathLike | onnx.ModelProto | qonnx ModelWrapper
        The quantised network.
    name : str, optional
        Prefix for the generated VHDL entities.  Defaults to the ONNX graph
        name, sanitised into a VHDL identifier.
    max_mult_bits : int
        Precision of the fixed-point requantisation multiplier used when a
        scale ratio is not a power of two (default 32).
    max_bias_frac_bits : int
        Maximum fractional accumulator bits used to represent a bias that is
        not an integer multiple of the accumulator scale (e.g. a float bias
        or a folded BatchNorm).  0 rounds such biases to the nearest integer.
    thresholds : "auto" | bool
        Implement the bias + [ReLU] + Quant after a linear / conv layer as
        per-channel integer thresholds on the accumulator (``2^bits - 1``
        comparisons, like FINN's MultiThreshold) instead of a fixed-point
        multiplier.  Thresholds are exact for any scale and bias (folded
        BatchNorm included).  ``"auto"`` (default) uses them when the
        multiplier form would be approximate and the output has at most
        ``AUTO_THRESHOLD_BITS`` (4) bits; ``True`` whenever the output has at
        most ``MAX_THRESHOLD_BITS`` (8) bits; ``False`` never.

    Raises
    ------
    UnsupportedOperatorError
        For operators / patterns without a hardware mapping.
    QONNXImportError
        For quantisation schemes that cannot be expressed in integer hardware.
    """
    mw = load_qonnx(model)
    out_name = mw.graph.output[0].name
    x = _network_input(mw)

    ir = IRGraph(
        name=vhdl_identifier(name or mw.graph.name or "network"),
        input_shape=x.onnx_shape,
        input_stream=x.stream,
        input_dtype=x.dtype,
        input_range=(x.lo, x.hi),
        input_scale=float(x.scale[0]),
        input_rounding=_input_rounding(mw),
    )
    b = _Builder(mw, ir, max_mult_bits, max_bias_frac_bits, thresholds)

    while x.name != out_name:
        node = _single_consumer(mw, x.name)
        if node is None:
            raise QONNXImportError(f"Tensor '{x.name}' is a dead end before the graph output.")
        op = node.op_type

        if op in _PASSTHROUGH_OPS:
            x = b.reshape(node, x)
        elif op in ("MatMul", "Gemm"):
            x = b.linear(node, x)
        elif op == "Conv":
            x = b.conv(node, x)
        elif op in ("MaxPool", "AveragePool"):
            x = b.pool(node, x)
        elif op in ("GlobalAveragePool", "GlobalMaxPool"):
            x = b.global_pool(node, x)
        elif op == "Pad":
            x = b.pad(node, x)
        elif op in _ELEMENTWISE_OPS:
            x = b.elementwise(node, x)
        else:
            raise UnsupportedOperatorError(op, node.name)

    if not ir.nodes:
        raise QONNXImportError("The model contains no computation to map to hardware.")

    ir.output_shape = x.onnx_shape
    ir.output_stream = x.stream
    ir.output_dtype = x.dtype
    ir.output_scale = np.tile(np.asarray(x.scale, dtype=np.float64), x.pixels)
    return ir
