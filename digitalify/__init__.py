"""
digitalify — ML-to-VHDL synthesis library.

Translates quantized neural-network models (expressed as operator-level
torch.fx graphs) into fully synthesisable structural VHDL descriptions.

Quick start — synthesis
-----------------------
>>> import torch, torch.nn as nn, torch.fx as fx, digitalify
>>> model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
>>> gm    = fx.symbolic_trace(model)
>>> files = digitalify.convert(gm, input_shape=(4,))

Quick start — GHDL simulation
------------------------------
>>> result = digitalify.simulate(gm, input_shape=(4,))
>>> print(result.summary())
[PASSED]  8 passed, 0 failed
"""

from digitalify.frontend.torch_importer import from_fx_graph
from digitalify.backend.vhdl.generator import VHDLGenerator
from digitalify.exceptions import UnsupportedOperatorError, VHDLGenerationError


def convert(
    graph_module,
    input_shape: tuple,
    data_width: int = 8,
    output_dir: str | None = None,
) -> dict[str, str]:
    """
    High-level entry point: trace → IR → VHDL.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The (possibly quantized) model after symbolic tracing.
    input_shape : tuple
        Shape of one input sample, e.g. ``(784,)`` or ``(4,)``.
    data_width : int
        Bit-width used for fixed-point arithmetic (default 8 → INT8).
    output_dir : str or None
        If given, generated VHDL files are also written to this directory.

    Returns
    -------
    dict[str, str]
        Mapping from filename (e.g. ``"linear_0.vhd"``) to VHDL source.
    """
    ir_graph = from_fx_graph(graph_module, input_shape=input_shape)
    generator = VHDLGenerator(data_width=data_width)
    files = generator.generate(ir_graph)

    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for fname, content in files.items():
            with open(os.path.join(output_dir, fname), "w") as fh:
                fh.write(content)

    return files


def simulate(
    graph_module,
    input_shape: tuple,
    data_width: int = 8,
    n_vectors: int = 8,
    seed: int = 42,
    ghdl_path: str = "ghdl",
    work_dir: str | None = None,
    keep_work_dir: bool = False,
    vcd: bool = True,
):
    """
    Full pipeline: trace → IR → VHDL → testbench → GHDL simulation.

    Generates random INT8 test vectors, computes expected outputs with the
    Python reference model, produces a self-checking VHDL testbench, and
    runs it through GHDL.  Requires GHDL ≥ 3.0 installed and on PATH
    (or pass its full path via ``ghdl_path``).

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced model.
    input_shape : tuple
        Shape of one input sample (batch dim excluded), e.g. ``(16,)``.
    data_width : int
        Activation bit-width (must match the synthesis step).
    n_vectors : int
        Number of test vectors (first three are always edge-cases).
    seed : int
        RNG seed for random test vectors.
    ghdl_path : str
        Name or full path of the GHDL executable (default: ``"ghdl"``).
    work_dir : str or None
        Directory for intermediate GHDL files.  A temporary directory is
        created and cleaned up automatically when ``None``.
    keep_work_dir : bool
        Keep the working directory after the run (useful for inspecting
        generated files or loading the VCD into GTKWave).
    vcd : bool
        Dump a VCD waveform to ``{work_dir}/waveform.vcd``.

    Returns
    -------
    GHDLResult
        Dataclass with ``passed``, ``n_passed``, ``n_failed``, ``log``,
        ``vcd_path``, and ``work_dir``.  Call ``.summary()`` for a
        one-line status string.
    """
    from pathlib import Path
    from digitalify.testing.tb_generator import generate_testbench
    from digitalify.testing.ghdl_runner import run_ghdl

    ir_graph = from_fx_graph(graph_module, input_shape=input_shape)
    vhdl_files = VHDLGenerator(data_width=data_width).generate(ir_graph)

    tb_filename, tb_source = generate_testbench(
        ir_graph,
        n_vectors=n_vectors,
        data_width=data_width,
        seed=seed,
    )
    tb_entity = f"tb_{ir_graph.name}"

    return run_ghdl(
        vhdl_files=vhdl_files,
        tb_filename=tb_filename,
        tb_source=tb_source,
        tb_entity=tb_entity,
        ghdl_path=ghdl_path,
        work_dir=Path(work_dir) if work_dir else None,
        keep_work_dir=keep_work_dir,
        vcd=vcd,
    )


__all__ = [
    "convert",
    "simulate",
    "from_fx_graph",
    "VHDLGenerator",
    "UnsupportedOperatorError",
    "VHDLGenerationError",
]
