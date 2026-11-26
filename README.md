# Digitalify

Convert quantized neural networks in **QONNX** format into fully synthesisable, **bit-exact** VHDL.

Digitalify reads a QONNX model — exported from Brevitas (`export_qonnx`), converted from QKeras, or built by hand — and generates structural VHDL-2008 that can go straight into any FPGA or ASIC synthesis flow (Vivado, Quartus, ...). It does not depend on PyTorch or any other training framework.

## How it works

The library is a classic compiler pipeline:

```
QONNX model (.onnx with Quant nodes)
      │
      ▼  digitalify.from_qonnx      qonnx cleanup + pattern matching
IRGraph  (integer IR: per-tensor bit widths, integer weights, requant constants)
      │
      ├─▶ digitalify.VHDLGenerator  one component per layer + structural top level
      ├─▶ testing.reference         bit-exact integer model of the hardware
      │
      ├─▶ digitalify.verify         integer model  ≡  QONNX execution (qonnx.core.onnx_exec)
      └─▶ digitalify.simulate       VHDL           ≡  integer model    (GHDL testbench)
```

`verify` and `simulate` together close the loop: the generated hardware computes exactly what the QONNX model computes.

### Integer semantics

For a layer with per-tensor input scale `s_x`, per-output-channel weight scale `s_w[j]` and output `Quant` scale `s_y[j]`:

```
acc[j] = Σ_i q_w[j,i] · q_x[i] · 2^F  +  bias[j]              (bias in units of s_x·s_w/2^F)
q_y[j] = clip( round( acc[j] · MULT[j] / 2^SHIFT[j] ) )        MULT/2^SHIFT = s_x·s_w[j]/s_y[j]
```

* Every float32 scale is a dyadic rational, so with power-of-two scales (the usual choice in Brevitas, QKeras, hls4ml) `MULT / 2^SHIFT` is **exact**. Arbitrary scales use a 32-bit fixed-point multiplier (`max_mult_bits`).
* Float (unquantised) biases — Brevitas' default — get `F` fractional accumulator bits (`max_bias_frac_bits`, default 16) instead of being rounded away.
* All seven QONNX rounding modes are implemented in hardware: `ROUND` (half-even), `HALF_UP`, `HALF_DOWN`, `FLOOR`, `CEIL`, `UP`, `DOWN`.
* Accumulator widths come from interval analysis of the actual weights and input range, so they never overflow and are no wider than needed.

## Supported patterns

| QONNX pattern | IR op | Hardware |
|---|---|---|
| `Quant` on the graph input (or an input with an integer QONNX DataType) | — | Sets the input bus type and scale |
| `MatMul` / `Gemm` with `Quant(W)` weights `[→ Add(bias)] [→ Relu] [→ Quant]` | `linear` | Combinational MAC, registered output with ReLU + requantisation (1 clock) |
| `Relu` on a quantised tensor | `relu` | Combinational, 0 clocks |
| `Flatten`, `Reshape`, `Identity`, `Squeeze`, `Unsqueeze` | — | Free (the bus is already flat) |

Weights can have any bit width and per-tensor or per-channel scales; each activation tensor has its own width and signedness (e.g. INT8 input → UINT4 hidden → INT8 output). If no `Quant` follows a layer, its accumulator is the output.

Anything else raises `UnsupportedOperatorError` (unknown op, branching graph, re-quantising an already quantised tensor, bipolar 1-bit quantisation) or `QONNXImportError` (unquantised input or weights, non-zero zero-point) at import time, before any VHDL is written. Convolutions and skip connections are the next operators on the roadmap.

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
#       export_qonnx(brevitas_model, torch.randn(1, 16), "model.onnx")
model = "model.onnx"

# 2 — inspect what will be built
ir = digitalify.from_qonnx(model)
print(ir.summary())

# 3 — generate VHDL
files = digitalify.convert(model, output_dir="vhdl_out/")

# 4 — check the integer hardware model against QONNX execution
print(digitalify.verify(model).summary())
# [PASSED]  64/64 vectors match QONNX, max error 0 LSB

# 5 — simulate the generated VHDL with GHDL
print(digitalify.simulate(model).summary())
# [PASSED]  16 passed, 0 failed
```

No model at hand? `digitalify.testing.make_mlp` builds QONNX MLPs (any widths, scales, rounding modes) with plain `onnx.helper`:

```python
from digitalify.testing import make_mlp
model = make_mlp([16, 8, 4], weight_bits=4, act_bits=6, output_bits=8)
```

See `examples/simple_network.ipynb` for a full walkthrough.

## Generated VHDL structure

Each layer becomes a self-contained entity; a structural top level wires them together. Entity names are prefixed with the network name, so several networks can share a VHDL library:

```
vhdl_out/
├── mlp_linear_0.vhd   # MatMul + bias + ReLU + requant, weights as constants
├── mlp_linear_1.vhd
└── mlp_top.vhd        # clk, rst, valid_i, data_i -> valid_o, data_o
```

Bus packing: element `i` of a bus with `B`-bit elements is `data((i+1)*B-1 downto i*B)`. A `valid` bit travels with the data; latency is 1 clock per `linear` layer (more with pipelining, below), and a new input vector can be accepted every clock.

The linear component, in outline:

```vhdl
-- Combinational MAC at the exact width the weights need:
-- products, then a balanced adder tree (depth ceil(log2(IN_F)))
for j in 0 to OUT_F - 1 loop
    for i in 0 to IN_F - 1 loop
        t(i) := resize(x(i) * WEIGHTS(j)(i), SUM_W);
    end loop;
    for k in 0 to 7 loop t(k) := t(2 * k) + t(2 * k + 1); end loop;   -- 16 -> 8
    ...                                                               -- 8 -> 4 -> 2 -> 1
    acc(j) <= shift_left(resize(t(0), ACC_W), F) + BIASES(j);
end loop;

-- Output register: ReLU, then q = clip(round(acc * MULT / 2^SHIFT))
p := resize(a * MULT(j), P_W);
q := shift_right(p, SHIFT(j));            -- floor
r := p and SHIFT_MASK(j);                 -- discarded bits drive the rounding
if r > half or (r = half and q(0) = '1') then q := q + 1; end if;   -- ROUND
```

### Pipelining

`convert(..., pipeline=...)` (also `simulate` and `VHDLGenerator(pipeline=...)`) adds optional registers inside every linear layer:

```
products -[products]- tree level 1 -[tree]- level 2 -[tree]- ... last level
    -> +bias -[acc]- ReLU -> x MULT -[mult]- round/clip -[out]-
```

| Stage | Register | Extra latency per layer |
|---|---|---|
| `"products"` | after the products x·w | 1 |
| `"tree"` / `"tree:K"` | inside the adder tree, every level / every K levels (never after the last level) | ⌈depth / K⌉ − 1, depth = ⌈log2(IN_F)⌉ |
| `"acc"` | after the last tree level + bias | 1 |
| `"mult"` | after the requantisation product (layers that requantise) | 1 |

`pipeline` takes a stage name, a list of names, or `True` for every stage with the tree registered at every level; the default (`None`) adds nothing, leaving one register per layer. Each register adds one clock of latency; throughput stays one vector per clock.

Measured on the 16→8→4 example (4-bit weights; Artix-7 xc7a200t, out-of-context, post-synthesis estimates):

| `pipeline` | Latency (cycles) | fmax | FFs | LUTs |
|---|---|---|---|---|
| `None` | 2 | 55 MHz | 82 | 1 522 |
| `["acc", "mult"]` | 6 | 62 MHz | 366 | 1 595 |
| `["tree:2", "acc"]` | 6 | 149 MHz | 682 | 1 546 |
| `["tree", "acc"]` | 9 | 230 MHz | 1 575 | 1 632 |
| `["products", "tree", "acc"]` | 11 | 258 MHz | 2 125 | 1 704 |
| `True` | 13 | 262 MHz | 2 263 | 1 744 |

The MAC (products, adder tree, bias) is the long path; once it is cut, the remaining limit is the ReLU + requantisation + rounding stage.

## Testing

```bash
pytest tests          # GHDL tests are skipped if GHDL is not found
```

The suite checks the rounding arithmetic, the importer's error handling and, for a matrix of models (mixed precision, odd widths, unsigned input, float bias, Gemm, per-channel and non-power-of-two scales, every rounding mode, ...), both `verify` (reference vs QONNX) and `simulate` (VHDL vs reference), the latter with and without pipelining. The generated testbench streams one vector per clock with periodic idle cycles and checks every clock that `valid_o` is high exactly when a result is due, so a wrong latency or a misaligned `valid` fails the test.

## Extending the library

Adding an operator takes four steps:

1. **Frontend** — map the QONNX pattern to an `IRNode` in `digitalify/frontend/qonnx_importer.py`.
2. **Hardware** — subclass `VHDLOperator` (`digitalify/operators/`) and implement `architecture(node, entity)`; set `latency` and `clocked` (override `node_latency(node)` if the latency depends on the node or on the `pipeline` option).
3. **Register** it in `digitalify/operators/registry.py`.
4. **Reference** — add its bit-exact model to `digitalify/testing/reference.py`, then a variant to `tests/test_end_to_end.py`.

## Design decisions

**Why QONNX?**
QONNX is framework-neutral: Brevitas, QKeras (via `qonnx.converters`) and hls4ml/FINN all speak it, and its `Quant` node carries everything the hardware needs — scale, bit width, signedness, narrow range and rounding mode — for every tensor. The `qonnx` package also provides the cleanup passes and a reference executor, which `verify` uses as the golden model.

**Why bit-exact integer arithmetic?**
The hardware should not just approximate the trained model: with power-of-two scales it reproduces the QONNX model's outputs exactly, which makes regressions trivially detectable.

**Why hard-coded weight constants?**
The design is self-contained: no external memory, DMA or boot sequence, and the synthesiser can constant-fold multiplications by small weights into LUTs. The fully parallel MAC trades area for latency; it suits small networks, and larger ones will need a folded/pipelined architecture.

## Project structure

```
digitalify/
├── __init__.py               # convert(), simulate(), verify()
├── exceptions.py             # UnsupportedOperatorError, QONNXImportError, VHDLGenerationError
├── quant.py                  # rounding modes, fixed-point requant multipliers
├── verify.py                 # integer model vs QONNX execution
├── ir/
│   ├── types.py              # IntType (bits, signedness)
│   ├── node.py               # IRNode
│   └── graph.py              # IRGraph (nodes, bus types, scales)
├── frontend/
│   └── qonnx_importer.py     # QONNX model → IRGraph
├── operators/
│   ├── base.py               # abstract VHDLOperator
│   ├── registry.py           # op_type → handler class
│   ├── linear.py             # LinearOperator
│   └── activations.py        # ReLUOperator
├── backend/vhdl/
│   ├── generator.py          # IRGraph → {filename: VHDL}
│   └── naming.py             # VHDL identifier sanitising
└── testing/
    ├── qonnx_builder.py      # make_mlp(): QONNX models without PyTorch
    ├── reference.py          # bit-exact integer model + test vectors
    ├── tb_generator.py       # self-checking VHDL testbench
    └── ghdl_runner.py        # GHDL analyse / elaborate / run

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
