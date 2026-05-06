"""
torch.fx → IRGraph importer.

Walks a ``torch.fx.GraphModule`` (produced by ``torch.fx.symbolic_trace``
or by the ``torch.ao.quantization`` pipeline) and converts each
computation node into an ``IRNode``.

Supported node categories
-------------------------
* ``call_module`` targeting ``nn.Linear``               → op_type ``"linear"``
* ``call_module`` targeting ``nn.quantized.Linear``     → op_type ``"quantized_linear"``
* ``call_module`` targeting ``nn.ReLU`` / ``nn.ReLU6`` → op_type ``"relu"``
* ``call_function`` / ``call_method`` for ``torch.relu``
  / ``F.relu``                                          → op_type ``"relu"``

All other node types (``placeholder``, ``get_attr``, ``output``) are handled
silently.  Any ``call_module`` or ``call_function`` node that does not match
a known operator raises ``UnsupportedOperatorError``.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx

from digitalify.ir.node import IRNode
from digitalify.ir.graph import IRGraph
from digitalify.exceptions import UnsupportedOperatorError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# torch functions / methods that map to relu
_RELU_FUNCTIONS = {
    torch.relu,
    torch.nn.functional.relu,
    torch.nn.functional.relu_,
}

_RELU_METHOD_NAMES = {"relu", "relu_"}


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach and convert a tensor to a numpy array (float32 for float tensors,
    int8 for quantized tensors)."""
    if t.is_quantized:
        return t.int_repr().numpy().astype(np.int8)
    return t.detach().float().numpy()


def _quantize_weight(w: np.ndarray) -> np.ndarray:
    """Round and clamp a float weight array to INT8, matching _int8_literal."""
    return np.clip(np.round(w), -128, 127).astype(np.int8)


def _quantize_bias(b: np.ndarray) -> np.ndarray:
    """Round and clamp a float bias array to INT32, matching _int32_literal."""
    lo, hi = -(2 ** 31), 2 ** 31 - 1
    return np.clip(np.round(b), lo, hi).astype(np.int32)


def _compute_shift(scale: float) -> int:
    """
    Approximate *scale* as 1 / 2^shift (right-shift amount).

    For a more accurate representation the caller can store the full float
    scale and let the backend decide.  This helper is a sensible default for
    power-of-two requantisation.
    """
    if scale <= 0 or scale >= 1:
        return 0
    return max(0, round(-math.log2(scale)))


# ---------------------------------------------------------------------------
# Module-type classifiers
# ---------------------------------------------------------------------------

def _is_linear(module) -> bool:
    return isinstance(module, (nn.Linear,))


def _is_quantized_linear(module) -> bool:
    # torch.ao / torch.nn.quantized path
    try:
        import torch.nn.quantized as nnq
        if isinstance(module, nnq.Linear):
            return True
    except Exception:
        pass
    # Dynamic-quantized linear (stored as a regular Linear with int8 weights
    # after quantize_dynamic) — detected by presence of _packed_params
    if hasattr(module, "_packed_params"):
        return True
    return False


def _is_relu(module) -> bool:
    return isinstance(module, (nn.ReLU, nn.ReLU6))


# ---------------------------------------------------------------------------
# Weight / bias extraction
# ---------------------------------------------------------------------------

def _extract_linear_attrs(module: nn.Linear) -> dict:
    weight = _quantize_weight(_tensor_to_numpy(module.weight))   # (out, in) int8
    bias = (
        _quantize_bias(_tensor_to_numpy(module.bias))
        if module.bias is not None else None
    )                                                             # (out,) int32 | None
    return {
        "weight": weight,
        "bias": bias,
        "scale": 1.0,
        "shift": 0,
        "in_zero": 0,
        "out_zero": 0,
    }


def _extract_quantized_linear_attrs(module) -> dict:
    """Extract weights and quantisation parameters from a quantized Linear."""
    # Works for both torch.nn.quantized.Linear and dynamic-quant variants
    try:
        weight_tensor, bias_tensor = module._packed_params._weight_bias()
    except AttributeError:
        # Fallback: hope the module exposes .weight() callable
        weight_tensor = module.weight()
        bias_tensor = module.bias()

    weight = _quantize_weight(_tensor_to_numpy(weight_tensor))  # int8
    bias = (
        _quantize_bias(_tensor_to_numpy(bias_tensor))
        if bias_tensor is not None else None
    )                                                            # int32 | None

    # Quantisation scale (output_scale / (input_scale * weight_scale))
    w_scale = (
        weight_tensor.q_scale()
        if weight_tensor.is_quantized
        else 1.0
    )
    out_scale = getattr(module, "scale", 1.0)
    in_scale = 1.0  # not always stored; use 1.0 as default
    combined_scale = (in_scale * w_scale) / max(out_scale, 1e-9)

    return {
        "weight": weight,
        "bias": bias,
        "scale": combined_scale,
        "shift": _compute_shift(combined_scale),
        "in_zero": getattr(module, "zero_point", 0),
        "out_zero": 0,
    }


# ---------------------------------------------------------------------------
# Main importer
# ---------------------------------------------------------------------------

def from_fx_graph(
    graph_module: fx.GraphModule,
    input_shape: tuple[int, ...],
) -> IRGraph:
    """
    Convert a ``torch.fx.GraphModule`` into an ``IRGraph``.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        A traced (and optionally quantized) model.
    input_shape : tuple[int, ...]
        Shape of one input sample, *without* the batch dimension.
        Example: ``(784,)`` for a flattened MNIST image, ``(4,)`` for a
        simple 4-feature MLP.

    Returns
    -------
    IRGraph
        Populated intermediate-representation graph ready for VHDL generation.

    Raises
    ------
    UnsupportedOperatorError
        If the graph contains a ``call_module`` or ``call_function`` node
        whose operator has no registered handler.
    """
    graph_module.eval()

    ir = IRGraph(
        name=type(graph_module).__name__.lower().replace("graphmodule", "network"),
        input_shape=input_shape,
    )

    # Map from fx node name → IRNode name (or "__input__" for placeholders)
    fx_name_to_ir: dict[str, str] = {}
    current_shape = input_shape
    node_counter: dict[str, int] = {}

    for fx_node in graph_module.graph.nodes:
        op = fx_node.op

        # ------------------------------------------------------------------ #
        # Graph bookkeeping nodes — not translated to VHDL components         #
        # ------------------------------------------------------------------ #
        if op == "placeholder":
            fx_name_to_ir[fx_node.name] = "__input__"
            continue

        if op in ("get_attr", "output"):
            # output node: record the final shape
            if op == "output" and fx_node.args:
                last_ir_name = fx_name_to_ir.get(
                    fx_node.args[0].name if hasattr(fx_node.args[0], "name") else "",
                    None,
                )
                if last_ir_name and last_ir_name != "__input__":
                    try:
                        ir.output_shape = ir.get_node(last_ir_name).output_shape
                    except KeyError:
                        ir.output_shape = current_shape
                else:
                    ir.output_shape = current_shape
            continue

        # ------------------------------------------------------------------ #
        # call_module — look up the actual nn.Module                          #
        # ------------------------------------------------------------------ #
        if op == "call_module":
            module = graph_module.get_submodule(fx_node.target)
            module_type_str = type(module).__qualname__

            # Determine input name(s) for this node
            input_ir_names = [
                fx_name_to_ir.get(
                    arg.name if hasattr(arg, "name") else str(arg), "__input__"
                )
                for arg in fx_node.args
                if hasattr(arg, "name")
            ]
            if not input_ir_names:
                input_ir_names = ["__input__"]

            # ---- Linear (float) ----------------------------------------- #
            if _is_linear(module):
                op_type = "linear"
                attrs = _extract_linear_attrs(module)
                out_shape = (module.out_features,)

            # ---- Linear (quantized) -------------------------------------- #
            elif _is_quantized_linear(module):
                op_type = "quantized_linear"
                attrs = _extract_quantized_linear_attrs(module)
                # out_features from weight shape
                out_features = attrs["weight"].shape[0]
                out_shape = (out_features,)

            # ---- ReLU ---------------------------------------------------- #
            elif _is_relu(module):
                op_type = "relu"
                attrs = {}
                out_shape = current_shape

            # ---- Unsupported --------------------------------------------- #
            else:
                raise UnsupportedOperatorError(
                    op_type=module_type_str,
                    node_name=fx_node.name,
                )

            # Build a unique IR name
            idx = node_counter.get(op_type, 0)
            node_counter[op_type] = idx + 1
            ir_name = f"{op_type}_{idx}"

            ir_node = IRNode(
                name=ir_name,
                op_type=op_type,
                input_names=input_ir_names,
                input_shape=current_shape,
                output_shape=out_shape,
                attrs=attrs,
            )
            ir.add_node(ir_node)
            fx_name_to_ir[fx_node.name] = ir_name
            current_shape = out_shape
            continue

        # ------------------------------------------------------------------ #
        # call_function / call_method — handle relu and raise on others       #
        # ------------------------------------------------------------------ #
        if op in ("call_function", "call_method"):
            target = fx_node.target
            is_relu = (
                target in _RELU_FUNCTIONS
                or (isinstance(target, str) and target in _RELU_METHOD_NAMES)
            )

            if is_relu:
                input_ir_names = [
                    fx_name_to_ir.get(
                        arg.name if hasattr(arg, "name") else str(arg), "__input__"
                    )
                    for arg in fx_node.args
                    if hasattr(arg, "name")
                ]
                if not input_ir_names:
                    input_ir_names = ["__input__"]

                op_type = "relu"
                idx = node_counter.get(op_type, 0)
                node_counter[op_type] = idx + 1
                ir_name = f"{op_type}_{idx}"

                ir_node = IRNode(
                    name=ir_name,
                    op_type=op_type,
                    input_names=input_ir_names,
                    input_shape=current_shape,
                    output_shape=current_shape,
                    attrs={},
                )
                ir.add_node(ir_node)
                fx_name_to_ir[fx_node.name] = ir_name
                continue

            # Unknown function/method
            target_str = (
                target.__name__
                if callable(target) and hasattr(target, "__name__")
                else str(target)
            )
            raise UnsupportedOperatorError(
                op_type=target_str,
                node_name=fx_node.name,
            )

    # If output_shape was never set (e.g. single-node graph)
    if not ir.output_shape:
        ir.output_shape = current_shape

    return ir
