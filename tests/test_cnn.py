"""
Streaming dataflow: convolutional networks, pooling, BatchNorm, lookup-table
activations, standalone requantisation and folded linear stages.

* verify   — integer reference model == QONNX execution
* simulate — generated VHDL == reference model, with random backpressure
"""

import numpy as np
import pytest
from onnx import helper

import digitalify
from digitalify import UnsupportedOperatorError
from digitalify.testing import make_cnn, make_mlp

CONV_POOL_LINEAR = [
    dict(type="conv", out=4, kernel=3, pad=1),
    dict(type="maxpool"),
    dict(type="conv", out=6, kernel=3, stride=2),
    dict(type="flatten"),
    dict(type="linear", out=5, relu=False, bits=8),
]

VARIANTS = {
    "conv_pool_linear": ((3, 8, 8), CONV_POOL_LINEAR),
    "flatten_image": ((2, 6, 6), [dict(type="conv", out=3, kernel=3), dict(type="flatten"),
                                  dict(type="linear", out=4, relu=False, bits=8)]),
    "batchnorm_gap": ((3, 8, 8), [dict(type="conv", out=4, kernel=3, pad=1, bn=True), dict(type="gap"),
                                  dict(type="flatten"), dict(type="linear", out=3, relu=False, bits=8)]),
    "depthwise_pointwise": ((4, 6, 6), [dict(type="conv", out=4, kernel=3, pad=1, groups=4),
                                        dict(type="conv", out=2, kernel=1, bits=8)]),
    "conv1d": ((3, 16), [dict(type="conv", out=4, kernel=3, pad=1), dict(type="maxpool"),
                         dict(type="conv", out=2, kernel=3, bits=8)]),
    "sigmoid_lut": ((2, 6, 6), [dict(type="conv", out=3, kernel=3, pad=1, bits=4, relu=False),
                                dict(type="act", fn="Sigmoid", bits=5, signed=False)]),
    "tanh_lut": ((2, 5, 5), [dict(type="conv", out=3, kernel=3, bits=4, relu=False),
                             dict(type="act", fn="Tanh", bits=4)]),
    "avgpool_requant": ((2, 8, 8), [dict(type="conv", out=3, kernel=3, pad=1), dict(type="avgpool"),
                                    dict(type="quant", bits=6, signed=False)]),
    "leaky_maxpool_pad": ((2, 6, 6), [dict(type="conv", out=3, kernel=3, relu=False, bits=5),
                                      dict(type="act", fn="LeakyRelu", bits=5),
                                      dict(type="maxpool", kernel=2, stride=1, pad=1)]),
    "stride_no_pad_raw_out": ((1, 7, 7), [dict(type="conv", out=2, kernel=3, stride=2, bits=None,
                                               relu=False)]),
    "unsigned_input": ((2, 6, 6), [dict(type="conv", out=3, kernel=2, stride=2, bits=6)]),
}


def _model(variant, **kw):
    shape, layers = VARIANTS[variant]
    return make_cnn(shape, layers, seed=1, input_signed=variant != "unsigned_input", **kw)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_reference_matches_qonnx(variant):
    result = digitalify.verify(_model(variant), n_vectors=12)
    assert result.passed, result.summary()


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_vhdl_matches_reference(variant, ghdl):
    result = digitalify.simulate(_model(variant), n_vectors=4, ghdl_path=ghdl)
    assert result.passed, result.summary() + "\n" + result.log[-3000:]
    assert result.n_passed == 4


def test_lowering_of_conv_pool_linear():
    ir = digitalify.from_qonnx(_model("conv_pool_linear"))
    assert [n.op_type for n in ir.nodes] == [
        "pad", "swg", "linear", "swg", "pool", "swg", "linear", "linear"]
    assert ir.input_stream == (8, 8, 3) and ir.output_stream == (1, 1, 5)
    conv = ir.nodes[2]
    assert conv.input_shape == (8, 8, 27) and conv.output_shape == (8, 8, 4)


def test_layout_round_trip():
    ir = digitalify.from_qonnx(_model("conv_pool_linear"))
    x = np.arange(2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    s = ir.onnx_to_stream(x)
    assert s.shape == (2, 192)
    assert s[0, :3].tolist() == [0, 64, 128]          # pixel (0, 0): channels 0..2
    np.testing.assert_array_equal(ir.stream_to_onnx(s), x)


def test_identical_lut_channels_share_a_table():
    ir = digitalify.from_qonnx(_model("sigmoid_lut"))
    lut = next(n for n in ir.nodes if n.op_type == "lut")
    assert len(lut.attrs["tables"]) == 1 and list(lut.attrs["table_of"]) == [0, 0, 0]


# ---- folding ----------------------------------------------------------------------

@pytest.mark.parametrize("folding", [
    {"linear_0": (2, 9)},
    {"linear_0": (1, 3), "linear_1": (3, 4), "linear_2": (5, 2)},
    {"linear_1": (1, 36)},
])
def test_folded_cnn(folding, ghdl):
    result = digitalify.simulate(_model("conv_pool_linear"), n_vectors=3, ghdl_path=ghdl,
                                 folding=folding)
    assert result.passed, result.summary() + "\n" + result.log[-3000:]


@pytest.mark.parametrize("folding", [(1, 1), (4, 2), (2, 16)])
def test_folded_mlp(folding, ghdl):
    model = make_mlp([16, 8, 4], seed=5)
    result = digitalify.simulate(model, n_vectors=6, ghdl_path=ghdl,
                                 folding={"linear_0": folding, "linear_1": (2, 4)})
    assert result.passed, result.summary() + "\n" + result.log[-3000:]


@pytest.mark.parametrize("pipeline", [True, "products", "tree:2", ["acc", "mult"], ["products", "tree"]])
@pytest.mark.parametrize("folding", [
    {"linear_0": (2, 9), "linear_1": (3, 4), "linear_2": (5, 2)},
    {"linear_0": (4, 1), "linear_1": (1, 1), "linear_2": (1, 6)},
])
def test_pipelined_folded_cnn(pipeline, folding, ghdl):
    result = digitalify.simulate(_model("conv_pool_linear"), n_vectors=3, ghdl_path=ghdl,
                                 folding=folding, pipeline=pipeline)
    assert result.passed, result.summary() + "\n" + result.log[-3000:]


@pytest.mark.parametrize("pipeline", [None, True])
def test_folded_mlp_issues_one_work_item_per_clock(pipeline, ghdl):
    """No bubble between beats: NF·SF clocks per beat, pipelined or not."""
    model = make_mlp([16, 8, 4], seed=5)
    runs = [digitalify.simulate(model, n_vectors=n, ghdl_path=ghdl, backpressure=False,
                                folding={"linear_0": (2, 4)}, pipeline=pipeline) for n in (4, 8)]
    assert all(r.passed for r in runs)
    assert runs[1].cycles - runs[0].cycles == 4 * (8 // 2) * (16 // 4)


def test_folding_trades_cycles_for_multipliers(ghdl):
    model = make_mlp([16, 8, 4], seed=5)
    full = digitalify.simulate(model, n_vectors=8, ghdl_path=ghdl, backpressure=False)
    folded = digitalify.simulate(model, n_vectors=8, ghdl_path=ghdl, backpressure=False,
                                 folding={"linear_0": (2, 4)})
    assert full.passed and folded.passed
    # 8 vectors: one per clock fully parallel, (8/2)·(16/4) = 16 clocks folded
    assert folded.cycles > full.cycles + 8 * 15
    files = digitalify.convert(model, folding={"linear_0": (2, 4)})
    assert "PE=2, SIMD=4, 16 clocks per beat, 8 multipliers" in files["mlp_top.vhd"]


def test_invalid_folding():
    model = make_mlp([16, 8, 4])
    with pytest.raises(ValueError, match="must divide"):
        digitalify.convert(model, folding={"linear_0": (3, 4)})
    with pytest.raises(ValueError, match="not linear stages"):
        digitalify.convert(model, folding={"conv_9": (1, 1)})


def test_full_throughput_one_beat_per_clock(ghdl):
    """Without backpressure a fully parallel network moves one pixel per clock."""
    model = _model("conv_pool_linear")
    runs = [digitalify.simulate(model, n_vectors=n, ghdl_path=ghdl, backpressure=False)
            for n in (4, 8)]
    assert all(r.passed for r in runs)
    # Steady state: the pad stage emits the 10x10 padded image (100 beats)
    # for every 64 input pixels, so each extra image costs 100 clocks.
    assert runs[1].cycles - runs[0].cycles == 4 * 100


# ---- unsupported patterns ---------------------------------------------------------

def _conv_attr_model(**attrs):
    model = _model("stride_no_pad_raw_out")
    conv = next(n for n in model.graph.node if n.op_type == "Conv")
    for k, v in attrs.items():
        conv.attribute.append(helper.make_attribute(k, v))
    return model


def test_dilation_is_rejected():
    with pytest.raises(UnsupportedOperatorError, match="Dilation"):
        digitalify.from_qonnx(_conv_attr_model(dilations=[2, 2]))


def test_avgpool_with_padding_is_rejected():
    model = make_cnn((2, 6, 6), [dict(type="conv", out=2, kernel=1),
                                 dict(type="avgpool", kernel=3, stride=1, pad=1),
                                 dict(type="quant", bits=6)])
    with pytest.raises(UnsupportedOperatorError, match="padding"):
        digitalify.from_qonnx(model)


def test_activation_without_quant_is_rejected():
    model = make_cnn((2, 6, 6), [dict(type="conv", out=2, kernel=1, relu=False, bits=8),
                                 dict(type="conv", out=2, kernel=3, relu=False, bits=None),
                                 dict(type="act", fn="Sigmoid", bits=4)])
    # the Sigmoid input is a raw accumulator: its range is too large for a table
    with pytest.raises(UnsupportedOperatorError, match="too large for a lookup table"):
        digitalify.from_qonnx(model)


# ---- pool tree --------------------------------------------------------------------

POOLS = {
    "max2x2": ((3, 8, 8), [dict(type="conv", out=4, kernel=3, pad=1), dict(type="maxpool")]),
    "max3x3": ((2, 9, 9), [dict(type="conv", out=3, kernel=3, pad=1),
                           dict(type="maxpool", kernel=3, stride=3)]),
    "avg3x3": ((2, 9, 9), [dict(type="conv", out=3, kernel=3, pad=1), dict(type="avgpool", kernel=3, stride=3),
                           dict(type="quant", bits=6, signed=False)]),
    "max1d": ((3, 16), [dict(type="conv", out=4, kernel=3, pad=1), dict(type="maxpool")]),
}


@pytest.mark.parametrize("pipeline", [None, "pool"])
@pytest.mark.parametrize("variant", sorted(POOLS))
def test_pool_tree_matches_reference(variant, pipeline, ghdl):
    shape, layers = POOLS[variant]
    model = make_cnn(shape, layers, seed=4)
    result = digitalify.simulate(model, n_vectors=3, ghdl_path=ghdl, pipeline=pipeline)
    assert result.passed, result.summary() + "\n" + result.log[-3000:]


@pytest.mark.parametrize("window,latency", [(1, 1), (2, 1), (4, 2), (9, 2)])
def test_pool_register_only_for_deep_trees(window, latency):
    from digitalify.operators.window import PoolOperator

    class N:                    # node_latency reads only the attributes
        attrs = {"window": window, "mode": "max"}
    assert PoolOperator("pool").node_latency(N) == latency
    assert PoolOperator(None).node_latency(N) == 1
