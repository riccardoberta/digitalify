"""
QONNX → IRGraph importer.

Accepts a QONNX model (file path, ``onnx.ModelProto`` or qonnx
``ModelWrapper``) as exported by Brevitas (``export_qonnx``), converted from
QKeras, or built by hand, and maps it to integer hardware.

The model is first normalised with ``qonnx.util.cleanup.cleanup_model``
(shape inference, constant folding with Quant nodes preserved, unique names,
batch size 1).  The graph is then walked from the single input to the single
output, matching these patterns on a chain of tensors:

    input ─► Quant                                   network input quantiser
          ─► Flatten / Reshape / Identity / ...      no-op on the flat bus
          ─► MatMul|Gemm(Quant(W)) [─► Add(b)] [─► Relu] [─► Quant]
                                                     → one ``linear`` node
          ─► Relu                                    → one ``relu`` node

Integer semantics
-----------------
With a per-tensor input scale ``s_x``, per-output-channel weight scale
``s_w[j]`` and output scale ``s_y[j]``:

    acc[j]   = Σ_i q_w[j,i]·q_x[i] + round(b[j] / (s_x·s_w[j]))
    q_y[j]   = clip(R(acc[j] · s_x·s_w[j] / s_y[j]))

If no ``Quant`` follows a layer, the (optionally ReLU'd) accumulator itself is
the layer output, with scale ``s_x·s_w[j]``.

Limitations (raise ``UnsupportedOperatorError`` / ``QONNXImportError``):
single input / output, no branching, zero-points must be 0, activation scales
must be per-tensor where they feed a MatMul, 1-bit bipolar quantisation and
Conv are not mapped yet.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
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
from digitalify.quant import normalize_rounding, requant_multiplier, to_fraction

# Ops that only change the logical shape of a flat bus.
_PASSTHROUGH_OPS = {"Flatten", "Reshape", "Identity", "Squeeze", "Unsqueeze"}

# Bound on accumulator width so the int64 reference model cannot overflow.
_MAX_ACC_BITS = 62


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


# ---------------------------------------------------------------------------
# Graph walking state
# ---------------------------------------------------------------------------

@dataclass
class _Tensor:
    """A quantised tensor flowing on the hardware bus."""
    name: str                 # ONNX tensor name
    shape: tuple[int, ...]    # batch dimension excluded
    dtype: IntType
    lo: int                   # tight integer range of the values
    hi: int
    scale: np.ndarray         # float64, one entry per element
    src: str                  # producing IR node name, or "__input__"

    @property
    def features(self) -> int:
        return int(prod(self.shape))


def _strip_batch(shape) -> tuple[int, ...]:
    shape = tuple(int(d) for d in (shape or ()))
    if len(shape) > 1 and shape[0] == 1:
        shape = shape[1:]
    return shape


def _single_consumer(mw, tensor: str):
    consumers = mw.find_consumers(tensor) or []
    if len(consumers) > 1:
        raise UnsupportedOperatorError(
            "branch", consumers[0].name,
            f"Tensor '{tensor}' feeds {len(consumers)} nodes; only chains are supported.",
        )
    return consumers[0] if consumers else None


# ---------------------------------------------------------------------------
# Pattern builders
# ---------------------------------------------------------------------------

def _network_input(mw) -> tuple[_Tensor, int]:
    """Quantised view of the graph input and the number of nodes consumed."""
    in_name = mw.graph.input[0].name
    shape = _strip_batch(mw.get_tensor_shape(in_name))
    n_feat = int(prod(shape))
    first = _single_consumer(mw, in_name)

    if first is not None and first.op_type in ("Quant", "BipolarQuant") and first.input[0] == in_name:
        qp = _quant_params(mw, first)
        if np.size(qp.scale) != 1:
            raise QONNXImportError(
                f"Input Quant '{first.name}': only a per-tensor input scale is supported."
            )
        t = _Tensor(first.output[0], shape, qp.dtype, qp.qmin, qp.qmax,
                    np.full(n_feat, float(qp.scale.reshape(-1)[0])), "__input__")
        return t, 1

    # FINN-style: input annotated with an integer DataType and no Quant node.
    dt = mw.get_tensor_datatype(in_name)
    if dt is not None and dt.is_integer() and dt.bitwidth() > 1:
        it = IntType(int(dt.bitwidth()), bool(dt.signed()))
        return _Tensor(in_name, shape, it, int(dt.min()), int(dt.max()),
                       np.ones(n_feat), "__input__"), 0

    raise QONNXImportError(
        "The network input must be quantised: the first node should be a Quant node "
        "(e.g. Brevitas QuantIdentity) or the input must carry an integer QONNX DataType."
    )


def _linear(mw, node, x: _Tensor, name: str, max_mult_bits: int, max_bias_frac_bits: int):
    """Fuse MatMul|Gemm [Add] [Relu] [Quant] into one ``linear`` IR node."""
    if np.any(x.scale != x.scale[0]):
        raise QONNXImportError(
            f"Node '{node.name}': the input has per-element scales; a MatMul input needs "
            "a single per-tensor scale (add a Quant node before it)."
        )

    bias_f = None
    if node.op_type == "MatMul":
        if node.input[0] != x.name:
            raise UnsupportedOperatorError("MatMul", node.name,
                                           "Only activation @ weights is supported.")
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

    out_f, in_f = q_w.shape
    if in_f != x.features:
        raise QONNXImportError(
            f"Node '{node.name}': weight matrix expects {in_f} inputs, bus carries {x.features}."
        )
    s_w_out = s_w[:, 0]
    if np.any(s_w != s_w_out[:, None]):
        raise QONNXImportError(
            f"Node '{node.name}': weight scales must be per-tensor or per output channel."
        )

    cur = node.output[0]
    nxt = _single_consumer(mw, cur)

    # ---- optional bias Add --------------------------------------------------
    if nxt is not None and nxt.op_type == "Add" and bias_f is None:
        other = nxt.input[1] if nxt.input[0] == cur else nxt.input[0]
        bf, bq = _constant(mw, other)
        if bf is not None:
            bias_f = _dequantized(bf, bq)
            cur, nxt = nxt.output[0], _single_consumer(mw, nxt.output[0])

    # ---- optional ReLU ------------------------------------------------------
    relu = False
    if nxt is not None and nxt.op_type == "Relu":
        relu = True
        cur, nxt = nxt.output[0], _single_consumer(mw, nxt.output[0])

    # ---- bias in accumulator units ---------------------------------------------
    # A bias that is not an integer multiple of s_x·s_w (e.g. a float bias,
    # Brevitas' default) gets F fractional bits: the accumulator becomes
    #   acc = (Σ w·x) · 2^F + round(b / (s_x·s_w) · 2^F)
    # and the requantisation factor is divided by 2^F.
    s_x = to_fraction(x.scale[0])
    acc_unit = [s_x * to_fraction(s) for s in s_w_out]
    bias = np.zeros(out_f, dtype=np.int64)
    frac_bits = 0
    bias_exact = True
    if bias_f is not None:
        b = _per_channel(bias_f, out_f, f"Bias of node '{node.name}'")
        v = [Fraction(float(b[j])) / acc_unit[j] for j in range(out_f)]
        while frac_bits < max_bias_frac_bits and any((vj * (1 << frac_bits)).denominator != 1 for vj in v):
            frac_bits += 1
        scaled = [vj * (1 << frac_bits) for vj in v]
        bias_exact = all(s.denominator == 1 for s in scaled)
        bias[:] = [round(s) for s in scaled]
    acc_unit = [u / (1 << frac_bits) for u in acc_unit]

    # ---- accumulator range ----------------------------------------------------
    w_dtype = IntType.for_range(int(q_w.min()), int(q_w.max()), signed=True)
    if x.dtype.bits + 1 + w_dtype.bits + int(in_f).bit_length() + frac_bits > _MAX_ACC_BITS:
        raise QONNXImportError(f"Node '{node.name}': accumulator would exceed {_MAX_ACC_BITS} bits.")
    t_lo, t_hi = q_w * x.lo, q_w * x.hi
    acc_hi = (np.maximum(t_lo, t_hi).sum(axis=1) << frac_bits) + bias
    acc_lo = (np.minimum(t_lo, t_hi).sum(axis=1) << frac_bits) + bias
    lo, hi = int(acc_lo.min()), int(acc_hi.max())
    # The MAC chain Σ w·x runs at its own (narrower) width; only the final
    # "sum·2^F + bias" adder is acc_bits wide.  Partial sums stay inside
    # [s_lo, s_hi] because every input range contains 0.
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

    # ---- optional output Quant (requantisation) -------------------------------
    requant = None
    if nxt is not None and nxt.op_type in ("Quant", "BipolarQuant") and nxt.input[0] == cur:
        qp = _quant_params(mw, nxt)
        s_y = _per_channel(qp.scale, out_f, f"Scale of Quant node '{nxt.name}'")
        mults, shifts, exact = [], [], True
        for j in range(out_f):
            m, n, ex = requant_multiplier(acc_unit[j] / to_fraction(s_y[j]), max_mult_bits)
            mults.append(m)
            shifts.append(n)
            exact &= ex
        requant = {
            "multiplier": np.array(mults, dtype=np.int64),
            "shift": np.array(shifts, dtype=np.int64),
            "rounding": normalize_rounding(qp.rounding),
            "clip_min": qp.qmin,
            "clip_max": qp.qmax,
            "exact": exact,
        }
        out_dtype = qp.dtype
        lo = max(qp.qmin, 0) if relu else qp.qmin
        hi = qp.qmax
        out_scale = s_y.astype(np.float64)
        cur = nxt.output[0]
    else:
        out_dtype = IntType.for_range(lo, hi)
        out_scale = np.array([float(u) for u in acc_unit])

    ir_node = IRNode(
        name=name,
        op_type="linear",
        input_names=[x.src],
        input_shape=x.shape,
        output_shape=(out_f,),
        input_dtype=x.dtype,
        output_dtype=out_dtype,
        attrs={
            "weight": q_w.astype(np.int64),
            "bias": bias,
            "bias_exact": bias_exact,
            "bias_frac_bits": frac_bits,
            "weight_dtype": w_dtype,
            "sum_dtype": IntType(sum_bits, True),
            "acc_dtype": acc_dtype,
            "relu": relu,
            "requant": requant,
            "onnx_node": node.name,
        },
    )
    y = _Tensor(cur, (out_f,), out_dtype, lo, hi, out_scale, name)
    return ir_node, y


def _per_channel(values, n: int, what: str) -> np.ndarray:
    """Flatten a scalar or per-channel constant to a length-*n* vector."""
    v = np.asarray(values).reshape(-1)
    if v.size == 1:
        return np.full(n, v[0], dtype=v.dtype)
    if v.size != n:
        raise QONNXImportError(f"{what}: expected 1 or {n} values, got shape {np.shape(values)}.")
    return v


def _dequantized(values: np.ndarray, qp: _QuantParams | None) -> np.ndarray:
    """Real value of a (possibly Quant-wrapped) constant, as qonnx computes it."""
    if qp is None:
        return values.astype(np.float32)
    return (qp.quantize(values).astype(np.float32) * qp.scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Main importer
# ---------------------------------------------------------------------------

def from_qonnx(
    model,
    name: str | None = None,
    max_mult_bits: int = 32,
    max_bias_frac_bits: int = 16,
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
        not an integer multiple of ``s_in * s_w`` (e.g. an unquantised float
        bias).  0 rounds such biases to the nearest integer.

    Raises
    ------
    UnsupportedOperatorError
        For operators / patterns without a hardware mapping.
    QONNXImportError
        For quantisation schemes that cannot be expressed in integer hardware.
    """
    mw = load_qonnx(model)

    out_name = mw.graph.output[0].name
    x, _ = _network_input(mw)

    ir = IRGraph(
        name=vhdl_identifier(name or mw.graph.name or "network"),
        input_shape=x.shape,
        input_dtype=x.dtype,
        input_range=(x.lo, x.hi),
        input_scale=float(x.scale[0]),
    )
    counters: dict[str, int] = {}

    def _new_name(op: str) -> str:
        idx = counters.get(op, 0)
        counters[op] = idx + 1
        return f"{op}_{idx}"

    while x.name != out_name:
        node = _single_consumer(mw, x.name)
        if node is None:
            raise QONNXImportError(f"Tensor '{x.name}' is a dead end before the graph output.")
        op = node.op_type

        if op in _PASSTHROUGH_OPS:
            new_shape = _strip_batch(mw.get_tensor_shape(node.output[0]))
            if int(prod(new_shape)) != x.features:
                raise QONNXImportError(f"Node '{node.name}' ({op}) changes the number of elements.")
            x = _Tensor(node.output[0], new_shape, x.dtype, x.lo, x.hi, x.scale, x.src)

        elif op in ("MatMul", "Gemm"):
            ir_node, x = _linear(mw, node, x, _new_name("linear"), max_mult_bits, max_bias_frac_bits)
            ir.add_node(ir_node)

        elif op == "Relu":
            nm = _new_name("relu")
            ir.add_node(IRNode(
                name=nm, op_type="relu", input_names=[x.src],
                input_shape=x.shape, output_shape=x.shape,
                input_dtype=x.dtype, output_dtype=x.dtype,
                attrs={"onnx_node": node.name},
            ))
            x = _Tensor(node.output[0], x.shape, x.dtype, max(x.lo, 0), max(x.hi, 0), x.scale, nm)

        elif op in ("Quant", "BipolarQuant"):
            raise UnsupportedOperatorError(
                op, node.name,
                "Re-quantising an already quantised tensor (outside a MatMul/Gemm block) "
                "is not mapped yet.",
            )

        else:
            raise UnsupportedOperatorError(op, node.name)

    if not ir.nodes:
        raise QONNXImportError("The model contains no computation to map to hardware.")

    ir.output_shape = x.shape
    ir.output_dtype = x.dtype
    ir.output_scale = np.asarray(x.scale, dtype=np.float64)
    return ir
