"""
End-to-end numerical check: integer hardware model vs. QONNX execution.

The VHDL testbench proves that the generated hardware matches the integer
reference model.  This module proves that the reference model matches the
*original* QONNX model, executed by ``qonnx.core.onnx_exec`` in floating
point.  Together they guarantee  hardware ≡ QONNX model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from digitalify.frontend.qonnx_importer import from_qonnx, load_qonnx
from digitalify.ir.graph import IRGraph
from digitalify.testing.reference import generate_test_vectors


@dataclass
class VerificationResult:
    """
    Attributes
    ----------
    n_vectors : int
    n_mismatch : int
        Output vectors with at least one element differing by more than the
        tolerance.
    max_error_lsb : float
        Largest |QONNX − reference| over all elements, in output LSBs.
    requant_exact : bool
        All requantisation factors are represented exactly (power-of-two
        scale ratios).  When False, rare ±1 LSB differences near rounding
        ties are expected.
    bias_exact : bool
        All biases are exact multiples of the accumulator scale.
    inputs, reference, golden : np.ndarray
        Integer inputs, reference outputs and QONNX outputs (in LSBs).
    """

    n_vectors: int
    n_mismatch: int
    max_error_lsb: float
    requant_exact: bool
    bias_exact: bool
    inputs: np.ndarray
    reference: np.ndarray
    golden: np.ndarray

    @property
    def passed(self) -> bool:
        return self.n_mismatch == 0

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        notes = []
        if not self.requant_exact:
            notes.append("approximate requantisation")
        if not self.bias_exact:
            notes.append("rounded bias")
        note = f"  ({', '.join(notes)})" if notes else ""
        return (f"[{status}]  {self.n_vectors - self.n_mismatch}/{self.n_vectors} vectors match "
                f"QONNX, max error {self.max_error_lsb:.3g} LSB{note}")


def verify(
    model,
    ir: IRGraph | None = None,
    n_vectors: int = 64,
    seed: int = 0,
    tolerance_lsb: float = 0.0,
) -> VerificationResult:
    """
    Compare the integer reference model of *model* with QONNX execution.

    Parameters
    ----------
    model : path | onnx.ModelProto | qonnx ModelWrapper
    ir : IRGraph, optional
        Pre-imported graph (imported from *model* when omitted).
    n_vectors : int
        Number of input vectors (edge cases + random).
    seed : int
        RNG seed.
    tolerance_lsb : float
        Allowed difference in output LSBs (use 1 for models with
        non-power-of-two scales).
    """
    from qonnx.core.onnx_exec import execute_onnx

    mw = load_qonnx(model)
    if ir is None:
        ir = from_qonnx(mw)

    inputs, reference = generate_test_vectors(ir, n_vectors=n_vectors, seed=seed)
    in_name = mw.graph.input[0].name
    out_name = mw.graph.output[0].name
    in_shape = mw.get_tensor_shape(in_name)

    golden = np.empty(reference.shape, dtype=np.float64)
    for k, q in enumerate(inputs):
        x = (q.astype(np.float64) * ir.input_scale).astype(np.float32).reshape(in_shape)
        y = execute_onnx(mw, {in_name: x})[out_name]
        golden[k] = np.asarray(y, dtype=np.float64).reshape(-1) / ir.output_scale

    err = np.abs(golden - reference)
    # QONNX runs in float32: allow its rounding noise (a few ulp of the largest
    # value seen on each output) on top of the requested tolerance.  For
    # requantised outputs (|value| < 2^bits) this stays far below half an LSB.
    float_noise = np.abs(golden).max(axis=0, keepdims=True) * 4e-6
    bad = err > tolerance_lsb + 1e-3 + float_noise
    linear = [n for n in ir.nodes if n.op_type == "linear"]
    return VerificationResult(
        n_vectors=len(inputs),
        n_mismatch=int(bad.any(axis=1).sum()),
        max_error_lsb=float(err.max()) if err.size else 0.0,
        requant_exact=all(n.attrs["requant"] is None or n.attrs["requant"]["exact"] for n in linear),
        bias_exact=all(n.attrs.get("bias_exact", True) for n in linear),
        inputs=inputs,
        reference=reference,
        golden=golden,
    )
