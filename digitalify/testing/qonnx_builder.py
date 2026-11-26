"""
Build small QONNX MLPs with ``onnx.helper`` — no PyTorch / Brevitas needed.

The graphs follow the structure Brevitas emits with ``export_qonnx``:

    global_in ─ Quant ─ MatMul(Quant(W)) ─ Add(b) ─ Relu ─ Quant ─ MatMul ... ─ global_out

Useful for tests, examples and for trying the flow before a real model exists.

>>> from digitalify.testing.qonnx_builder import make_mlp
>>> model = make_mlp([16, 8, 4], weight_bits=4, act_bits=6)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

QONNX_DOMAIN = "qonnx.custom_op.general"


def _per_layer(v, n: int) -> list:
    if isinstance(v, Sequence) and not isinstance(v, str):
        if len(v) != n:
            raise ValueError(f"expected {n} values, got {len(v)}")
        return list(v)
    return [v] * n


def make_mlp(
    layer_sizes: Sequence[int],
    *,
    input_bits: int = 8,
    input_signed: bool = True,
    weight_bits: int | Sequence[int] = 8,
    act_bits: int | Sequence[int] = 8,
    output_bits: int = 8,
    output_quant: bool = True,
    relu: bool = True,
    bias: bool = True,
    quantized_bias: bool = True,
    per_channel: bool = False,
    pow2_scales: bool = True,
    use_gemm: bool = False,
    rounding_mode: str = "ROUND",
    input_shape: Sequence[int] | None = None,
    seed: int = 0,
    name: str = "mlp",
) -> onnx.ModelProto:
    """
    Build a quantised multi-layer perceptron as a QONNX ``ModelProto``.

    Parameters
    ----------
    layer_sizes : [in, hidden..., out]
    input_bits, input_signed : input quantiser.
    weight_bits : per-model or per-layer weight bit-width (signed, narrow range).
    act_bits : per-model or per-hidden-layer activation bit-width.  Hidden
        activations are unsigned when ``relu`` is True.
    output_bits, output_quant : quantiser on the final layer (signed); with
        ``output_quant=False`` the last layer outputs its raw accumulator.
    relu : ReLU after every hidden layer.
    bias, quantized_bias : add a bias; when ``quantized_bias`` it is an exact
        multiple of ``s_in * s_w`` (Brevitas ``Int32Bias`` style).
    per_channel : per-output-channel weight scales.
    pow2_scales : power-of-two scales (bit-exact hardware) vs arbitrary floats.
    use_gemm : emit ``Gemm`` (transB=1, bias as C) instead of ``MatMul``+``Add``.
    rounding_mode : rounding mode of the activation / output Quant nodes.
    input_shape : optional multi-dimensional input shape (batch excluded) whose
        product equals ``layer_sizes[0]``; a ``Flatten`` node is inserted.
    """
    rng = np.random.default_rng(seed)
    n_layers = len(layer_sizes) - 1
    w_bits = _per_layer(weight_bits, n_layers)
    a_bits = _per_layer(act_bits, max(n_layers - 1, 1))

    nodes, inits = [], []

    def const(name, value, dtype=np.float32):
        inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def scale_value(shape=()):
        if pow2_scales:
            v = 2.0 ** -rng.integers(2, 7, size=shape)
        else:
            v = rng.uniform(0.01, 0.2, size=shape)
        return np.asarray(v, dtype=np.float32)

    def quant(inp, out, scale, bits, signed, narrow, rounding="ROUND"):
        tag = out
        nodes.append(helper.make_node(
            "Quant",
            [inp, const(f"{tag}_scale", scale), const(f"{tag}_zeropt", 0.0),
             const(f"{tag}_bitwidth", float(bits))],
            [out], domain=QONNX_DOMAIN, name=f"Quant_{tag}",
            signed=int(signed), narrow=int(narrow), rounding_mode=rounding,
        ))
        return out

    in_dims = list(input_shape) if input_shape is not None else [layer_sizes[0]]
    if int(np.prod(in_dims)) != layer_sizes[0]:
        raise ValueError("input_shape does not match layer_sizes[0]")

    x = "global_in"
    s_x = scale_value()
    x = quant(x, "inp_q", s_x, input_bits, input_signed, False)
    if input_shape is not None and len(in_dims) > 1:
        nodes.append(helper.make_node("Flatten", [x], ["inp_flat"], axis=1, name="Flatten_0"))
        x = "inp_flat"

    for li in range(n_layers):
        fin, fout = layer_sizes[li], layer_sizes[li + 1]
        last = li == n_layers - 1

        # Weights: float values, quantised by a Quant node (signed, narrow).
        s_w = scale_value((1, fout) if per_channel else ())
        w = rng.normal(0.0, 0.5, size=(fin, fout)).astype(np.float32) * (2 ** (w_bits[li] - 2)) * s_w
        if use_gemm:   # Gemm with transB=1 takes B as (out, in)
            wq = quant(const(f"W{li}", w.T), f"W{li}_q", s_w.T if per_channel else s_w,
                       w_bits[li], True, True)
        else:
            wq = quant(const(f"W{li}", w), f"W{li}_q", s_w, w_bits[li], True, True)

        b_name = None
        if bias:
            acc_scale = (s_x * s_w).reshape(-1)
            if quantized_bias:
                b = rng.integers(-64, 64, size=fout) * acc_scale
            else:
                b = rng.normal(0.0, 1.0, size=fout) * 16 * acc_scale
            b_name = const(f"b{li}", b.astype(np.float32))

        y = f"mm{li}"
        if use_gemm:
            ins = [x, wq] + ([b_name] if b_name else [])
            nodes.append(helper.make_node("Gemm", ins, [y], transB=1, name=f"Gemm_{li}"))
        else:
            nodes.append(helper.make_node("MatMul", [x, wq], [y], name=f"MatMul_{li}"))
            if b_name:
                nodes.append(helper.make_node("Add", [y, b_name], [f"add{li}"], name=f"Add_{li}"))
                y = f"add{li}"

        if not last:
            if relu:
                nodes.append(helper.make_node("Relu", [y], [f"relu{li}"], name=f"Relu_{li}"))
                y = f"relu{li}"
            s_a = scale_value()
            x = quant(y, f"act{li}_q", s_a, a_bits[li], not relu, False, rounding_mode)
            s_x = s_a
        else:
            if output_quant:
                quant(y, "global_out", scale_value(), output_bits, True, False, rounding_mode)
            else:
                nodes.append(helper.make_node("Identity", [y], ["global_out"], name="Identity_out"))

    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [1, *in_dims])],
        [helper.make_tensor_value_info("global_out", TensorProto.FLOAT, [1, layer_sizes[-1]])],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(QONNX_DOMAIN, 1)],
        producer_name="digitalify.testing.qonnx_builder",
    )
    model.ir_version = 8
    return model


def make_cnn(
    input_shape: Sequence[int],
    layers: Sequence[dict],
    *,
    input_bits: int = 8,
    input_signed: bool = True,
    weight_bits: int = 4,
    act_bits: int = 6,
    seed: int = 0,
    name: str = "cnn",
) -> onnx.ModelProto:
    """
    Build a quantised network over an image ``(C, H, W)`` or a sequence
    ``(C, L)`` as a QONNX ``ModelProto`` (NCHW, as Brevitas exports it).

    Each entry of *layers* is a dict with a ``type`` key:

    * ``conv``: ``out``, ``kernel`` (3), ``stride`` (1), ``pad`` (0), ``groups``
      (1), ``bias`` (True), ``bn`` (False: BatchNormalization after the conv),
      ``relu`` (True), ``bits`` (output Quant bits; ``None`` = no Quant).
    * ``linear``: ``out``, ``bias``, ``relu``, ``bits`` - on a flat tensor.
    * ``maxpool`` / ``avgpool``: ``kernel`` (2), ``stride`` (= kernel), ``pad`` (0).
    * ``gap``: GlobalAveragePool.
    * ``flatten``: Flatten (axis 1).
    * ``act``: ``fn`` (ONNX op, e.g. "Sigmoid", "Tanh", "LeakyRelu"), ``bits``,
      ``signed`` (True): activation followed by a Quant.
    * ``quant``: ``bits``, ``signed``, ``scale``: a standalone Quant.

    Scales are powers of two unless BatchNorm or average pooling make the
    arithmetic non-dyadic.
    """
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    counter = [0]

    def const(name, value):
        inits.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return name

    def uid(prefix):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def pow2(lo=2, hi=6):
        return np.float32(2.0 ** -int(rng.integers(lo, hi + 1)))

    def quant(inp, scale, bits, signed, narrow=False, out=None):
        out = out or uid("q")
        nodes.append(helper.make_node(
            "Quant", [inp, const(f"{out}_s", scale), const(f"{out}_z", 0.0),
                      const(f"{out}_b", float(bits))],
            [out], domain=QONNX_DOMAIN, name=f"Quant_{out}",
            signed=int(signed), narrow=int(narrow), rounding_mode="ROUND"))
        return out

    shape = list(input_shape)              # (C, *spatial)
    x = quant("global_in", pow2(), input_bits, input_signed)

    for li, spec in enumerate(layers):
        kind = spec["type"]
        last = li == len(layers) - 1
        out_name = "global_out" if last else None
        if kind == "conv":
            c_in, nd = shape[0], len(shape) - 1
            c_out, groups = spec["out"], spec.get("groups", 1)
            k, st, pd = spec.get("kernel", 3), spec.get("stride", 1), spec.get("pad", 0)
            s_w = pow2(3, 6)
            w = rng.integers(-(2 ** (weight_bits - 1)) + 1, 2 ** (weight_bits - 1),
                             size=(c_out, c_in // groups, *([k] * nd))) * s_w
            wq = quant(const(uid("W"), w), s_w, weight_bits, True, narrow=True)
            ins = [x, wq]
            if spec.get("bias", True):
                ins.append(const(uid("b"), rng.integers(-32, 32, size=c_out) * s_w * 2.0 ** -3))
            y = uid("conv")
            nodes.append(helper.make_node(
                "Conv", ins, [y], name=f"Conv_{y}", kernel_shape=[k] * nd, strides=[st] * nd,
                pads=[pd] * (2 * nd), group=groups))
            shape = [c_out] + [(d + 2 * pd - k) // st + 1 for d in shape[1:]]
            x = y
            if spec.get("bn"):
                bn = [const(uid("bn"), v) for v in (
                    rng.uniform(0.5, 1.5, c_out), rng.uniform(-0.5, 0.5, c_out),
                    rng.uniform(-0.5, 0.5, c_out), rng.uniform(0.5, 2.0, c_out))]
                y = uid("bn")
                nodes.append(helper.make_node("BatchNormalization", [x, *bn], [y], name=f"BN_{y}",
                                              epsilon=1e-5))
                x = y
        elif kind == "linear":
            fin = int(np.prod(shape))
            fout = spec["out"]
            s_w = pow2(3, 6)
            w = rng.integers(-(2 ** (weight_bits - 1)) + 1, 2 ** (weight_bits - 1), size=(fin, fout)) * s_w
            wq = quant(const(uid("W"), w), s_w, weight_bits, True, narrow=True)
            y = uid("mm")
            nodes.append(helper.make_node("MatMul", [x, wq], [y], name=f"MatMul_{y}"))
            x = y
            if spec.get("bias", True):
                y = uid("add")
                nodes.append(helper.make_node(
                    "Add", [x, const(uid("b"), rng.integers(-32, 32, size=fout) * s_w * 2.0 ** -3)],
                    [y], name=f"Add_{y}"))
                x = y
            shape = [fout]
        elif kind in ("maxpool", "avgpool"):
            nd = len(shape) - 1
            k = spec.get("kernel", 2)
            st = spec.get("stride", k)
            pd = spec.get("pad", 0)
            y = uid("pool")
            nodes.append(helper.make_node(
                "MaxPool" if kind == "maxpool" else "AveragePool", [x], [y], name=f"Pool_{y}",
                kernel_shape=[k] * nd, strides=[st] * nd, pads=[pd] * (2 * nd)))
            shape = [shape[0]] + [(d + 2 * pd - k) // st + 1 for d in shape[1:]]
            x = y
        elif kind == "gap":
            y = uid("gap")
            nodes.append(helper.make_node("GlobalAveragePool", [x], [y], name=f"GAP_{y}"))
            shape = [shape[0]] + [1] * (len(shape) - 1)
            x = y
        elif kind == "flatten":
            y = uid("flat")
            nodes.append(helper.make_node("Flatten", [x], [y], axis=1, name=f"Flatten_{y}"))
            shape = [int(np.prod(shape))]
            x = y
        elif kind == "act":
            y = uid("act")
            attrs = {"alpha": 0.125} if spec["fn"] == "LeakyRelu" else {}
            nodes.append(helper.make_node(spec["fn"], [x], [y], name=f"Act_{y}", **attrs))
            x = quant(y, pow2(2, 5), spec.get("bits", act_bits), spec.get("signed", True), out=out_name)
            continue
        elif kind == "quant":
            x = quant(x, np.float32(spec.get("scale", pow2(1, 4))), spec.get("bits", act_bits),
                      spec.get("signed", True), out=out_name)
            continue
        else:
            raise ValueError(f"unknown layer type {kind!r}")

        if kind in ("conv", "linear"):
            relu = spec.get("relu", True)
            if relu:
                y = uid("relu")
                nodes.append(helper.make_node("Relu", [x], [y], name=f"Relu_{y}"))
                x = y
            bits = spec.get("bits", act_bits)
            if bits is not None:
                x = quant(x, pow2(1, 4), bits, not relu, out=out_name)
                continue
        if last:
            nodes.append(helper.make_node("Identity", [x], ["global_out"], name="Identity_out"))

    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [1, *input_shape])],
        [helper.make_tensor_value_info("global_out", TensorProto.FLOAT, [1, *shape])],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(QONNX_DOMAIN, 1)],
        producer_name="digitalify.testing.qonnx_builder",
    )
    model.ir_version = 8
    return model
