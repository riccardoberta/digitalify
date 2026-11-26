"""Optional pipeline registers inside linear layers."""

import pytest

import digitalify
from digitalify import VHDLGenerator
from digitalify.operators.base import normalize_pipeline
from digitalify.operators.linear import LinearOperator
from digitalify.testing import make_mlp
from digitalify.testing.ghdl_runner import run_ghdl
from digitalify.testing.tb_generator import generate_testbench


@pytest.fixture(scope="module")
def ir():
    return digitalify.from_qonnx(make_mlp([16, 8, 4], seed=3))


ALL = {"products", "tree", "acc", "mult"}


@pytest.mark.parametrize("value,stages,every", [
    (None, set(), 1), (False, set(), 1), (True, ALL, 1),
    ("acc", {"acc"}, 1), (["mult"], {"mult"}, 1), (("acc", "mult"), {"acc", "mult"}, 1),
    ("tree", {"tree"}, 1), (["products", "tree:3"], {"products", "tree"}, 3),
])
def test_normalize_pipeline(value, stages, every):
    p = normalize_pipeline(value)
    assert p.stages == stages and p.tree_every == every
    assert normalize_pipeline(p) is p


@pytest.mark.parametrize("bad", [["acc", "adder"], "fast", "acc:2", "tree:0", "tree:x"])
def test_unknown_stage_rejected(bad):
    with pytest.raises(ValueError):
        normalize_pipeline(bad)


def test_convert_rejects_bad_pipeline():
    with pytest.raises(ValueError):
        digitalify.convert(make_mlp([4, 2]), pipeline="fast")


# 16 -> 8 -> 4: tree depth 4 (16 inputs) and 3 (8 inputs); both layers requantise.
# Core latency of the two linear stages (clocks from input beat to output register).
@pytest.mark.parametrize("pipeline,latency", [
    (None, 2), ("acc", 4), ("mult", 4), ("products", 4),
    ("tree", 2 + 3 + 2), ("tree:2", 2 + 1 + 1), ("tree:4", 2),
    (True, 7 + 6),
])
def test_latency(ir, pipeline, latency):
    op = LinearOperator(pipeline)
    assert sum(op.node_latency(n) for n in ir.nodes) == latency
    top = VHDLGenerator(pipeline).generate(ir)[f"{ir.name}_top.vhd"]
    assert ("-- Pipeline:" in top) == (pipeline is not None)


def test_single_input_layer_has_no_tree_registers():
    assert LinearOperator("tree")._mac_cuts(1) == []
    assert LinearOperator(True)._mac_cuts(1) == [0]


def test_mult_stage_needs_requantisation():
    raw = digitalify.from_qonnx(make_mlp([4, 3, 2], output_quant=False, seed=0))
    last = raw.nodes[-1]
    assert last.attrs.get("requant") is None
    assert LinearOperator("mult").node_latency(last) == 1
    # 3 inputs: out + products + 1 tree register (depth 2) + acc; no mult
    assert LinearOperator(True).node_latency(last) == 4


def test_registers_only_when_enabled(ir):
    plain = digitalify.convert(ir)
    piped = digitalify.convert(ir, pipeline=True)
    layer = f"{ir.name}_linear_0.vhd"
    assert "acc_reg" not in plain[layer] and "mult_reg" not in plain[layer]
    assert "acc_reg" in piped[layer] and "mult_reg" in piped[layer]
    assert "mac_reg0" in piped[layer] and "mac_reg" not in plain[layer]


@pytest.mark.parametrize("n,depth", [(1, 0), (2, 1), (5, 3), (16, 4), (17, 5)])
def test_adder_tree_depth(n, depth):
    tree = LinearOperator._adder_tree(n)
    assert tree.count("loop t(k)") == depth


@pytest.mark.parametrize("n", [1, 3, 4, 7, 16, 33])
def test_adder_tree_sums_everything(n):
    """Run the generated reduction on Python lists."""
    t = list(range(1, n + 1))
    for line in LinearOperator._adder_tree(n).splitlines():
        line = line.strip()
        if line.startswith("for k in 0 to"):
            hi = int(line.split()[5])          # for k in 0 to <hi> loop ...
            for k in range(hi + 1):
                t[k] = t[2 * k] + t[2 * k + 1]
        else:                                   # t(a) := t(b);
            a, b = (int(x.split("(")[1].split(")")[0]) for x in line.split(":="))
            t[a] = t[b]
    assert t[0] == n * (n + 1) // 2


def test_testbench_detects_wrong_output(ir, ghdl):
    """A testbench expecting a different first output beat must fail."""
    vhdl = VHDLGenerator(True).generate(ir)
    tb_name, tb_src = generate_testbench(ir, n_vectors=6)
    head, tail = tb_src.split("constant EXPECTED : out_array_t := (", 1)
    first = tail.index('"') + 1
    flipped = "1" if tail[first] == "0" else "0"
    tb_src = head + "constant EXPECTED : out_array_t := (" + tail[:first] + flipped + tail[first + 1:]
    result = run_ghdl(vhdl_files=vhdl, tb_filename=tb_name, tb_source=tb_src,
                      tb_entity=f"tb_{ir.name}", ghdl_path=ghdl)
    assert not result.passed
    assert result.n_failed == 1 and result.n_passed == 5
