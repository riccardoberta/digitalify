"""
digitalify — QONNX-to-VHDL synthesis library.

Translates quantised neural networks expressed in QONNX (as exported by
Brevitas ``export_qonnx``, converted from QKeras, or built by hand) into
fully synthesisable, bit-exact structural VHDL.

Quick start
-----------
>>> import digitalify
>>> files  = digitalify.convert("model.onnx", output_dir="vhdl_out/")
>>> check  = digitalify.verify("model.onnx")      # integer model vs QONNX
>>> sim    = digitalify.simulate("model.onnx")    # VHDL vs integer model (GHDL)
>>> print(check.summary(), sim.summary(), sep="\\n")
"""

from __future__ import annotations

from digitalify.backend.vhdl.generator import VHDLGenerator
from digitalify.exceptions import (
    QONNXImportError,
    UnsupportedOperatorError,
    VHDLGenerationError,
)
from digitalify.frontend.qonnx_importer import from_qonnx, load_qonnx
from digitalify.ir import IntType, IRGraph, IRNode
from digitalify.verify import VerificationResult, verify


def convert(model, output_dir: str | None = None, name: str | None = None,
            pipeline=None) -> dict[str, str]:
    """
    High-level entry point: QONNX → IR → VHDL.

    Parameters
    ----------
    model : str | os.PathLike | onnx.ModelProto | qonnx ModelWrapper | IRGraph
        The quantised network (or an already imported IRGraph).
    output_dir : str or None
        If given, the VHDL files are also written to this directory.
    name : str or None
        Prefix for the VHDL entities (defaults to the ONNX graph name).
    pipeline : bool | str | iterable of str, optional
        Extra registers inside every linear layer: ``"products"`` (after
        the products), ``"tree"`` / ``"tree:K"`` (inside the adder tree,
        every level / every K levels), ``"acc"`` (after the adder tree +
        bias), ``"mult"`` (after the requantisation product); ``True`` for
        all of them.  Default: none (one register per layer).  Each
        register adds one clock of latency and shortens the critical path.

    Returns
    -------
    dict[str, str]
        ``{filename: vhdl_source}``; the top level is ``<name>_top.vhd``.
    """
    ir = model if isinstance(model, IRGraph) else from_qonnx(model, name=name)
    files = VHDLGenerator(pipeline).generate(ir)

    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for fname, content in files.items():
            with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as fh:
                fh.write(content)
    return files


def simulate(
    model,
    n_vectors: int = 16,
    seed: int = 42,
    ghdl_path: str | None = None,
    work_dir: str | None = None,
    keep_work_dir: bool = False,
    vcd: bool = False,
    name: str | None = None,
    pipeline=None,
):
    """
    QONNX → VHDL → self-checking testbench → GHDL simulation.

    The testbench compares the VHDL against the bit-exact integer reference
    model; use :func:`verify` to compare that reference with the original
    QONNX model.  ``pipeline`` has the same meaning as in :func:`convert`.
    Requires GHDL ≥ 3.0 (on PATH, in ``DIGITALIFY_GHDL``, or passed as
    ``ghdl_path``).

    Returns
    -------
    GHDLResult
        ``passed``, ``n_passed``, ``n_failed``, ``log``, ``vcd_path``,
        ``work_dir``; ``.summary()`` gives a one-line status.
    """
    from pathlib import Path

    from digitalify.testing.ghdl_runner import run_ghdl
    from digitalify.testing.tb_generator import generate_testbench

    ir = model if isinstance(model, IRGraph) else from_qonnx(model, name=name)
    vhdl_files = VHDLGenerator(pipeline).generate(ir)
    tb_filename, tb_source = generate_testbench(ir, n_vectors=n_vectors, seed=seed,
                                                pipeline=pipeline)

    return run_ghdl(
        vhdl_files=vhdl_files,
        tb_filename=tb_filename,
        tb_source=tb_source,
        tb_entity=f"tb_{ir.name}",
        ghdl_path=ghdl_path,
        work_dir=Path(work_dir) if work_dir else None,
        keep_work_dir=keep_work_dir,
        vcd=vcd,
    )


__all__ = [
    "convert",
    "simulate",
    "verify",
    "from_qonnx",
    "load_qonnx",
    "VHDLGenerator",
    "IRGraph",
    "IRNode",
    "IntType",
    "VerificationResult",
    "UnsupportedOperatorError",
    "QONNXImportError",
    "VHDLGenerationError",
]
