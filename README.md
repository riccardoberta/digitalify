# Digitalify

Convert quantized neural networks in **QONNX** format into fully synthesisable, **bit-exact**, streaming VHDL.

Digitalify reads a QONNX model — exported from Brevitas (`export_qonnx`), converted from QKeras, or built by hand — and generates structural VHDL-2008 that can go straight into any FPGA or ASIC synthesis flow (Vivado, Quartus, ...). MLPs and convolutional networks (1-D and 2-D) are supported. It does not depend on PyTorch or any other training framework.

## How it works

The library is a classic compiler pipeline:

```
QONNX model (.onnx with Quant nodes)
      │
      ▼  digitalify.from_qonnx      qonnx cleanup + lowering to streaming stages
IRGraph  (integer IR: stream shapes, bit widths, integer weights, requant constants)
      │
      ├─▶ digitalify.VHDLGenerator  one component per stage + structural top level
      ├─▶ testing.reference         bit-exact integer model of the hardware
      │
      ├─▶ digitalify.verify         integer model  ≡  QONNX execution (qonnx.core.onnx_exec)
      └─▶ digitalify.simulate       VHDL           ≡  integer model    (GHDL testbench)
```

`verify` and `simulate` together close the loop: the generated hardware computes exactly what the QONNX model computes.

The hardware is a **dataflow pipeline**: every tensor travels as a stream of beats with a valid/ready handshake, and every layer is a stage that can stall. The design rationale is in [docs/dataflow_design.md](docs/dataflow_design.md).

### Integer semantics

For a layer with per-tensor input scale `s_x`, per-output-channel weight scale `s_w[j]` and output `Quant` scale `s_y[j]`:

```
acc[j] = Σ_i q_w[j,i] · q_x[i] · 2^F  +  bias[j]              (bias in units of s_x·s_w/2^F)
q_y[j] = clip( round( acc[j] · MULT[j] / 2^SHIFT[j] ) )        MULT/2^SHIFT = s_x·s_w[j]/s_y[j]
```

* Every float32 scale is a dyadic rational, so with power-of-two scales (the usual choice in Brevitas, QKeras, hls4ml) `MULT / 2^SHIFT` is **exact**. Arbitrary scales use a 32-bit fixed-point multiplier (`max_mult_bits`).
* Float (unquantised) biases — Brevitas' default — and folded BatchNorm offsets get `F` fractional accumulator bits (`max_bias_frac_bits`, default 16) instead of being rounded away.
* All seven QONNX rounding modes are implemented in hardware: `ROUND` (half-even), `HALF_UP`, `HALF_DOWN`, `FLOOR`, `CEIL`, `UP`, `DOWN`.

**Threshold activations.** The output quantiser is monotone in `acc`.  So for a `B`-bit output, bias + [ReLU] + `Quant` can instead be `2^B − 1` integer thresholds per channel:

```
q_y[j] = BASE[j] + #{k : Σ_i q_w[j,i]·q_x[i]  ≥  T[j][k]}
```

* The thresholds are computed with exact rational arithmetic from the float32 parameters, for any rounding mode.  So the result is exact for **any** scale and bias, including non-power-of-two scales and folded BatchNorm.
* There is no multiplier and no fractional bias bits.  The hardware is one comparator per threshold, plus a binary search over the resulting thermometer code.
* `from_qonnx(model, thresholds="auto")` (the default) uses them when the multiplier form would be approximate and the output has at most 4 bits.  `thresholds=True` uses them for outputs of up to 8 bits; `False` never uses them.
* Accumulator widths come from interval analysis of the actual weights and input range, so they never overflow and are no wider than needed.

## Supported patterns

| QONNX pattern | Hardware stages |
|---|---|
| `Quant` on the graph input (or an input with an integer QONNX DataType) | Sets the input stream type and scale |
| `MatMul` / `Gemm` with `Quant(W)` weights | `linear` |
| `Conv` 1-D / 2-D: any kernel, stride, zero padding, groups (depthwise) | `[pad] → swg → linear` |
| … each followed by `[Add / Sub / Mul / Div by constants, BatchNormalization]… [Relu] [Quant]` | fused into the `linear` stage |
| `MaxPool` (with padding) / `AveragePool` (no padding) | `[pad] → swg → pool` |
| `GlobalAveragePool` | `global_pool` |
| `Pad` (constant zero) | `pad` |
| `Flatten` / `Reshape` of an image into a vector | `flatten` (weights of the next layer permuted) |
| `Relu` | `relu` |
| `[Relu] Quant` on a quantised tensor | `requant` |
| Any chain of elementwise ops ending in `Quant` (`Sigmoid`, `Tanh`, `Clip`, `LeakyRelu`, `HardSigmoid`, `Elu`, `Mul`/`Add` by constants, …) | `lut` — a table computed by running the chain with qonnx, so it matches QONNX by construction (input range ≤ 4096 values) |
| `Flatten`, `Reshape`, `Identity`, `Squeeze`, `Unsqueeze` on vectors | free |

Weights can have any bit width and per-tensor or per-channel scales; each activation tensor has its own width and signedness (e.g. INT8 input → UINT4 hidden → INT8 output). If no `Quant` follows a layer, its accumulator is the output.

Anything else raises `UnsupportedOperatorError` (unknown op, branching graph such as residual connections, dilation, bipolar 1-bit quantisation, ...) or `QONNXImportError` (unquantised input or weights, non-zero zero-point) at import time, before any VHDL is written. Residual connections and AXI4-Stream wrappers are next on the roadmap.

## Installation

```bash
pip install -e .            # numpy, onnx, qonnx
pip install -e ".[dev]"     # + pytest, jupyter
```

Requires Python ≥ 3.10. Simulation additionally needs [GHDL](https://github.com/ghdl/ghdl/releases) ≥ 3.0, either on `PATH` or pointed to by the `DIGITALIFY_GHDL` environment variable.

## Quick start

```python
import digitalify

# 1 — a QONNX model, e.g. from Brevitas:
#       from brevitas.export import export_qonnx
#       export_qonnx(brevitas_model, torch.randn(1, 3, 32, 32), "model.onnx")
model = "model.onnx"

# 2 — inspect what will be built
ir = digitalify.from_qonnx(model)
print(ir.summary())

# 3 — generate VHDL (optionally folding the large layers, see below)
files = digitalify.convert(model, output_dir="vhdl_out/")

# 4 — check the integer hardware model against QONNX execution
print(digitalify.verify(model).summary())
# [PASSED]  64/64 vectors match QONNX, max error 0 LSB

# 5 — simulate the generated VHDL with GHDL (random backpressure on both ends)
print(digitalify.simulate(model).summary())
# [PASSED]  16 passed, 0 failed, 1752 cycles (first output after 368)
```

No model at hand? `digitalify.testing` builds QONNX models with plain `onnx.helper`:

```python
from digitalify.testing import make_cnn, make_mlp

mlp = make_mlp([16, 8, 4], weight_bits=4, act_bits=6, output_bits=8)
cnn = make_cnn((3, 32, 32), [
    dict(type="conv", out=8, kernel=3, pad=1),          # + ReLU + 6-bit Quant
    dict(type="maxpool"),
    dict(type="conv", out=16, kernel=3, pad=1, bn=True),
    dict(type="gap"),
    dict(type="flatten"),
    dict(type="linear", out=10, relu=False, bits=8),
])
```

See `examples/simple_network.ipynb` for a full walkthrough.

### A real model: MNIST CNN from Brevitas

`examples/mnist` trains a small Brevitas CNN and runs it through the whole flow:
8-bit input, two 3×3 convolutions with BatchNorm, 4-bit ReLU and max pooling,
then a 400 → 10 linear layer.

```bash
pip install -e ".[examples]"
python examples/mnist/train_brevitas.py    # ~1 min on a CPU, writes mnist_cnn.onnx
python examples/mnist/run_mnist.py
```

```
integer model accuracy: 0.9843 on 10000 test images      (Brevitas: 0.9843)
integer model vs QONNX: [PASSED]  200/200 vectors match QONNX
VHDL vs integer model:  [PASSED]  10 passed, 0 failed
throughput: 784 clocks per image, latency 746 clocks
```

`ir.quantize_input(images)` applies the network's input `Quant` to real-valued
images.  `verify` and `simulate` accept the result as `inputs=`.

Vivado 2025.1 out-of-context synthesis, Artix-7 XC7A200T, `pipeline=True`, 784 clocks per image:

| Configuration | Activation | LUT | FF | DSP | fmax | Power | Images/s |
|---|---|---|---|---|---|---|---|
| fully parallel | multiplier (`thresholds=False`) | 55,139 | 77,920 | 136 | 83 MHz | 3.5 W | 106 k |
| fully parallel | thresholds on the 4-bit layers (`"auto"`, default) | 53,895 | 77,386 | 40 | 88 MHz | 1.6 W | 112 k |
| fully parallel | thresholds everywhere (`True`) | 76,206 | 79,622 | 0 | 83 MHz | 3.9 W | 106 k |
| folded `linear_1` (16, 24), `linear_2` (2, 40) | `"auto"` | 17,883 | 13,404 | 8 | 18 MHz | 0.8 W | 23 k |

All configurations use 1.5 BRAM (the line buffers).

* With the default thresholds, the two BatchNorm layers need no multiplier at
  all.  96 DSPs go away, power halves, and the requantisation is exact.
* Fully parallel with `"auto"`, the critical path is the requantisation of the
  final 8-bit layer: a 32-bit multiplier, then rounding.
* `thresholds=True` also turns that layer into 255 thresholds per output.
  This removes every DSP but costs 22 k LUTs, so for 8-bit outputs the
  multiplier is the better choice.
* Folded, the critical path is the folded `linear` stage, which is not pipelined
  yet: ROM → 40 multipliers → adder tree → accumulator.

## Generated hardware

### Streams and handshake

A tensor of shape `(C, H, W)` travels as `H·W` beats in raster order, one pixel (all `C` channels) per beat; a vector is a single beat. Inside a beat, element `i` with `B` bits is `data((i+1)*B-1 downto i*B)`. ONNX's NCHW layout is converted at the boundaries (`ir.onnx_to_stream` / `ir.stream_to_onnx`).

Every stage and the top level have the same ports:

```
clk, rst
in_valid,  in_ready,  in_data       -- a beat moves when valid and ready are both '1'
out_valid, out_ready, out_data      -- (AXI4-Stream rules)
```

Each stage is a core that advances on a clock enable, followed by a two-register skid buffer whose registered `in_ready` is that enable. `out_ready` therefore never reaches `in_ready` combinationally: back-pressure costs no timing, however deep the network.

```
vhdl_out/
├── cnn_skid.vhd       # shared skid buffer
├── cnn_pad_0.vhd      # zero border
├── cnn_swg_0.vhd      # sliding window generator (line buffers)
├── cnn_linear_0.vhd   # conv as a matrix-vector unit, weights as constants
├── ...
└── cnn_top.vhd        # stages wired with valid / ready / data
```

### Stages

| Stage | Function | Rate |
|---|---|---|
| `linear` | `requant(relu(W·x + b))` on every beat | 1 beat/clock; folded: 1 beat per `(out/PE)·(in/SIMD) + 1` clocks |
| `pad` | border pixels of a constant value | emits `Hp·Wp` beats per `H·W` |
| `swg` | `KH-1` line buffers + window registers → one `KH·KW·C` window per output position | 1 pixel/clock |
| `pool` | per-channel max / sum over a window | 1/clock |
| `global_pool` | per-channel sum over all pixels | 1 pixel/clock |
| `flatten` | collects all pixels into one beat | 1 pixel/clock |
| `relu`, `requant`, `lut` | elementwise | 1/clock |

A convolution is `[pad] → swg → linear`: the window beat feeds the same matrix-vector unit as a dense layer, with the weights reshaped to `(C_out, KH·KW·C_in)`. Grouped and depthwise convolutions use a block-sparse weight matrix whose zeros the synthesiser removes in fully parallel mode.

### Folding: area vs throughput

By default a `linear` stage is fully parallel: one multiplier per weight, one beat per clock. `folding={"stage": (PE, SIMD)}` computes `PE` neurons × `SIMD` inputs per clock instead, with the weights in a ROM:

```python
files = digitalify.convert(model, folding={"linear_0": (2, 3), "linear_1": (4, 8)})
```

`PE` must divide the outputs and `SIMD` the inputs of the stage; stage names are listed by `ir.summary()`. Measured on the 32×32×3 CNN above (three convolutions, Artix-7 xc7a200t, out-of-context, post-synthesis):

| Configuration | LUTs | FFs | BRAM tiles | fmax | Power |
|---|---|---|---|---|---|
| Fully parallel | 49 249 | 8 738 | 5.5 | 13 MHz | 2.79 W |
| Fully parallel, `pipeline=True` | 59 799 | 74 514 | 5.5 | 132 MHz | 4.73 W |
| Folded (`linear_0` 2×3, `linear_1`/`linear_2` 4×8, `linear_3` 2×4) | 8 172 | 11 231 | 5.5 | 55 MHz | 0.52 W |

The line buffers map to block RAM. With full pipelining the critical path moves to the max-pool stage (a sequential compare over the window), which a comparison tree would shorten. In the folded unit the critical path runs from the weight-ROM address through the multiplier and the SIMD adder tree into the accumulator; registering the ROM output is the next improvement.

### Pipelining

`convert(..., pipeline=...)` (also `simulate` and `VHDLGenerator(pipeline=...)`) adds optional registers inside every fully parallel linear stage:

```
products -[products]- tree level 1 -[tree]- level 2 -[tree]- ... last level
    -> +bias -[acc]- ReLU -> x MULT -[mult]- round/clip -[out]-
```

| Stage | Register | Extra latency per layer |
|---|---|---|
| `"products"` | after the products x·w | 1 |
| `"tree"` / `"tree:K"` | inside the adder tree, every level / every K levels (never after the last level) | ⌈depth / K⌉ − 1, depth = ⌈log2(IN_F)⌉ |
| `"acc"` | after the last tree level + bias | 1 |
| `"mult"` | after the requantisation product, or the threshold comparisons (layers that requantise) | 1 |

`pipeline` takes a stage name, a list of names, or `True` for every stage with the tree registered at every level; the default (`None`) adds nothing. Each register adds one clock of latency; throughput stays one beat per clock.

Measured on the 16→8→4 MLP (4-bit weights; Artix-7 xc7a200t, out-of-context, post-synthesis estimates, before the streaming interface was added):

| `pipeline` | Latency (cycles) | fmax | FFs | LUTs |
|---|---|---|---|---|
| `None` | 2 | 55 MHz | 82 | 1 522 |
| `["acc", "mult"]` | 6 | 62 MHz | 366 | 1 595 |
| `["tree:2", "acc"]` | 6 | 149 MHz | 682 | 1 546 |
| `["tree", "acc"]` | 9 | 230 MHz | 1 575 | 1 632 |
| `["products", "tree", "acc"]` | 11 | 258 MHz | 2 125 | 1 704 |
| `True` | 13 | 262 MHz | 2 263 | 1 744 |

## Testing

```bash
pytest tests          # GHDL tests are skipped if GHDL is not found
```

The suite checks the rounding arithmetic, the importer's error handling and, for a matrix of MLPs (mixed precision, odd widths, unsigned input, float bias, Gemm, per-channel and non-power-of-two scales, every rounding mode, ...) and CNNs (padding, stride, max / average / global pooling, BatchNorm, depthwise, 1-D convolution, lookup-table activations, flatten of an image, ...), both `verify` (reference vs QONNX) and `simulate` (VHDL vs reference), with pipelining and folding.

The generated testbench streams every beat of several test tensors back to back while throttling `in_valid` and `out_ready` pseudo-randomly. It compares every output beat in order, checks that a stalled output holds its valid and data, fails on a timeout, and reports the cycle count (`GHDLResult.cycles`, `first_output`). With `simulate(..., backpressure=False)` the cycle count measures full throughput.

## Extending the library

Adding a stage takes four steps:

1. **Frontend** — lower the QONNX pattern to `IRNode`s in `digitalify/frontend/qonnx_importer.py`.
2. **Hardware** — subclass `VHDLOperator` (`digitalify/operators/`) and implement `core(node)`: logic that advances when `ce = '1'`, drives `in_ready` and the output register `core_valid` / `core_data`. The base class adds the ports and the skid buffer.
3. **Register** it in `digitalify/operators/registry.py`.
4. **Reference** — add its bit-exact model to `digitalify/testing/reference.py`, then a variant to `tests/test_cnn.py` or `tests/test_end_to_end.py`.

## Design decisions

**Why QONNX?**
QONNX is framework-neutral: Brevitas, QKeras (via `qonnx.converters`) and hls4ml/FINN all speak it, and its `Quant` node carries everything the hardware needs — scale, bit width, signedness, narrow range and rounding mode — for every tensor. The `qonnx` package also provides the cleanup passes and a reference executor, which `verify` uses as the golden model.

**Why bit-exact integer arithmetic?**
The hardware should not just approximate the trained model: with power-of-two scales it reproduces the QONNX model's outputs exactly, which makes regressions trivially detectable.

**Why streams with valid / ready?**
Layers run at different rates (pooling, folded units, line buffers filling up), so every stage must be able to stall. The handshake is the AXI4-Stream / Avalon-ST one, which keeps vendor wrappers thin, and the registered skid buffers keep the ready path short.

**Why hard-coded weight constants?**
The design is self-contained: no external memory, DMA or boot sequence, and the synthesiser can constant-fold multiplications by small weights into LUTs. Folding moves the weights into ROMs when a layer is too large to unroll.

## Project structure

```
digitalify/
├── __init__.py               # convert(), simulate(), verify()
├── exceptions.py             # UnsupportedOperatorError, QONNXImportError, VHDLGenerationError
├── quant.py                  # rounding modes, fixed-point requant multipliers
├── verify.py                 # integer model vs QONNX execution
├── ir/
│   ├── types.py              # IntType (bits, signedness)
│   ├── node.py               # IRNode (stream shapes)
│   └── graph.py              # IRGraph (nodes, stream types, scales, layout conversion)
├── frontend/
│   └── qonnx_importer.py     # QONNX model → IRGraph (lowering to stages)
├── operators/
│   ├── base.py               # VHDLOperator, ports, skid buffer, pipeline option
│   ├── registry.py           # op_type → handler class
│   ├── linear.py             # fully parallel and folded matrix-vector unit
│   ├── window.py             # pad, swg, pool, global_pool, flatten
│   └── activations.py        # relu, lut, requant
├── backend/vhdl/
│   ├── generator.py          # IRGraph → {filename: VHDL}
│   └── naming.py             # VHDL identifier sanitising
└── testing/
    ├── qonnx_builder.py      # make_mlp(), make_cnn(): QONNX models without PyTorch
    ├── reference.py          # bit-exact integer model + test vectors
    ├── tb_generator.py       # self-checking streaming VHDL testbench
    └── ghdl_runner.py        # GHDL analyse / elaborate / run

docs/dataflow_design.md       # architecture of the streaming hardware
tests/                        # pytest suite
examples/simple_network.ipynb # end-to-end walkthrough
vivado_api/                   # REST service for remote Vivado synthesis (below)
setup.py
```

---

## Vivado API

A REST API that exposes a remote Vivado installation as a service. It accepts VHDL files (including those generated by `digitalify`), runs synthesis and simulation on a chosen FPGA board, and returns structured performance reports.

### Setup

```bash
cd vivado_api
pip install -r requirements.txt
cp .env.example .env          # then edit .env for your server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://<server>:8000/docs` once the server is running.

### Configuration

All settings are read from environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `VIVADO_API_VIVADO_PATH` | `C:/Xilinx/2025.1/Vivado/bin/vivado.bat` (Windows), `/tools/Xilinx/2025.1/Vivado/bin/vivado` (Linux) | Path to the Vivado binary |
| `VIVADO_API_WORKSPACE_DIR` | `C:/vivado_api` (Windows), `/tmp/vivado_api` (Linux) | Root directory for uploads and job artefacts |
| `VIVADO_API_API_KEY` | *(empty)* | Bearer token for auth; empty disables auth |
| `VIVADO_API_SYNTHESIS_TIMEOUT` | `3600` | Max seconds for a synthesis run |
| `VIVADO_API_SIMULATION_TIMEOUT` | `600` | Max seconds for a simulation run |

### Available boards

| Board ID | Board | Family | LUTs |
|---|---|---|---|
| `basys3` | Digilent Basys3 | Artix-7 | 33 280 |
| `nexys4ddr` | Digilent Nexys4 DDR | Artix-7 | 101 400 |
| `nexys_video` | Digilent Nexys Video | Artix-7 | 215 360 |
| `pynq_z2` | TUL PYNQ-Z2 | Zynq-7000 | 53 200 |
| `zedboard` | Avnet ZedBoard | Zynq-7000 | 53 200 |
| `kcu105` | Xilinx KCU105 | Kintex UltraScale | 242 400 |
| `vcu118` | Xilinx VCU118 | Virtex UltraScale+ | 1 182 240 |
| `kv260` | AMD Kria KV260 Vision AI Starter Kit | Zynq UltraScale+ | 117 120 |

### Typical workflow

```
1.  GET  /boards                       → pick a board_id

2.  POST /upload  (multipart files)    → get file_id tokens for each .vhd

3.  POST /jobs    {board_id, top_entity, file_ids}
                                       → get job_id  (status: pending)

4.  POST /jobs/{id}/synthesize         → trigger Vivado  (status: synthesizing)
         {clock_period_ns, clock_port, out_of_context, strategy, flatten_hierarchy}
    GET  /jobs/{id}                    → poll until synthesized or failed

5.  GET  /jobs/{id}/synthesis          → utilization, timing, power + log tail
    GET  /jobs/{id}/log?type=synthesis → full Vivado log
    GET  /jobs/{id}/reports            → full parsed reports + raw text

6.  POST /upload  (testbench file)     → get testbench file_id
    POST /jobs/{id}/simulate  {testbench_file_id, testbench_top, runtime}
                                       → trigger xsim  (status: simulating)
    GET  /jobs/{id}                    → poll until done or failed
    GET  /jobs/{id}/simulation         → log tail and elapsed time
```

### Synthesis options

`POST /jobs/{id}/synthesize` accepts an optional JSON body:

| Field | Default | Meaning |
|---|---|---|
| `clock_period_ns` | *none* | Creates a clock on `clock_port` and a 0 ns input/output delay on every other port, so the full input → MAC → register path is timed. **Without it Vivado has no constraints and the timing / power reports are not meaningful.** |
| `clock_port` | `clk` | Clock port of the top entity (digitalify uses `clk`). |
| `out_of_context` | `false` | `synth_design -mode out_of_context`: no I/O buffers. Use it for digitalify networks, whose ports are not tied to board pins — otherwise every bus bit becomes an IOB and power is estimated for unconstrained pins. |
| `strategy` | `default` | `synth_design -directive` value. |
| `flatten_hierarchy` | `rebuilt` | `rebuilt`, `full` or `none`. |

Constraint files (`.xdc`) uploaded with the sources are also read; use them instead of `clock_period_ns` for real board pin-outs.

### Reports returned

After synthesis completes, three structured reports are available:

**Utilization** — LUT, flip-flop, DSP, BRAM counts with percentage of device used.

**Timing** — worst-case setup slack (WNS), total negative slack (TNS), hold slack (WHS/THS), a boolean `timing_met` (setup), the `clock_period_ns` used and the implied `fmax_mhz` = 1000 / (period − WNS). These are post-synthesis estimates; hold is only meaningful after implementation.

**Power** — total on-chip power, split into dynamic and static components, plus junction temperature and confidence level.

### End-to-end with digitalify

```python
import digitalify, requests

# Generate VHDL locally
files = digitalify.convert("model.onnx")

BASE = "http://my-server:8000"
HDR  = {"Authorization": "Bearer my-secret-token"}

# Upload all generated files
file_ids = []
for fname, vhdl in files.items():
    r = requests.post(f"{BASE}/upload", headers=HDR,
                      files={"files": (fname, vhdl.encode())})
    file_ids += [f["file_id"] for f in r.json()["files"]]

# Create job and synthesize
job = requests.post(f"{BASE}/jobs", headers=HDR, json={
    "board_id": "nexys4ddr",
    "top_entity": f"{digitalify.from_qonnx('model.onnx').name}_top",
    "file_ids": file_ids,
}).json()

requests.post(f"{BASE}/jobs/{job['job_id']}/synthesize", headers=HDR,
              json={"clock_period_ns": 10, "out_of_context": True})

# Poll until done, then fetch reports
import time
while True:
    status = requests.get(f"{BASE}/jobs/{job['job_id']}", headers=HDR).json()["status"]
    if status in ("synthesized", "failed"):
        break
    time.sleep(5)

reports = requests.get(f"{BASE}/jobs/{job['job_id']}/reports", headers=HDR).json()
print("LUTs used :", reports["utilization"]["luts"])
print("Timing met:", reports["timing"]["timing_met"], " fmax:", reports["timing"]["fmax_mhz"], "MHz")
print("Power (W) :", reports["power"]["total_power_w"])
```

---

## License

MIT
