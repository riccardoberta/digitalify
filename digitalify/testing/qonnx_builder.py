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
