# Digitalify

Convert quantized PyTorch models into fully synthesisable VHDL descriptions.

Digitalify takes a neural network (expressed as a torch.fx operator-level graph after quantization) and generates structural VHDL that can be fed directly into any standard FPGA or ASIC synthesis flow (Vivado, Quartus, ...)).

## How it works

The library is organized as a classic compiler pipeline:

```
PyTorch model
      │
      ▼  torch.fx.symbolic_trace
torch.fx GraphModule  (operator-level graph)
      │
      ▼  digitalify.frontend.from_fx_graph
IRGraph  (technology-independent IR)
      │
      ▼  digitalify.backend.VHDLGenerator
VHDL files  (one component per layer + one top-level structural entity)
```

Weights and biases are embedded as VHDL `constant` arrays, so the synthesiser sees them as ROM/LUT constants without any external memory interface. Activations flow through the design as `std_logic_vector` buses; a `valid` handshake signal propagates through the chain to mark when data is present.

## Supported operators

| Operator | op_type | VHDL behaviour |
|---|---|---|
| `nn.Linear` | `linear` | Combinational INT8 MAC array, 32-bit accumulator, registered output with saturation |
| `nn.quantized.Linear` | `quantized_linear` | Same as above; quantization scale/shift baked in |
| `nn.ReLU`, `nn.ReLU6`, `F.relu` | `relu` | Fully combinational comparator, zero latency |

Any layer not in this table raises `UnsupportedOperatorError` immediately, so you always know exactly what is missing before attempting synthesis.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

## Quick start

```python
import torch
import torch.nn as nn
import torch.fx as fx
import digitalify

# 1 — define and quantize a model
model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
model.eval()

quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 2 — trace to operator-level graph
gm = fx.symbolic_trace(quantized)

# 3 — generate VHDL
files = digitalify.convert(gm, input_shape=(16,), output_dir="vhdl_out/")

for name in files:
    print(name)
# linear_0.vhd
# relu_0.vhd
# linear_1.vhd
# network_top.vhd
```

See `examples/simple_network.ipynb` for a full walkthrough including model definition, quantization, tracing, VHDL generation, and error handling.

## Generated VHDL structure

Each layer produces a self-contained entity + architecture file. A top-level structural entity wires them together:

```
vhdl_out/
├── linear_0.vhd       # first Linear layer (entity + architecture, weights embedded)
├── relu_0.vhd         # ReLU activation
├── linear_1.vhd       # second Linear layer
└── network_top.vhd    # structural top-level, connects all components in order
```

The linear component architecture looks like this at a high level:

```vhdl
-- Combinational MAC (weights are VHDL constants)
for j in 0 to OUT_FEATURES-1 loop
    temp_acc := BIASES(j);
    for i in 0 to IN_FEATURES-1 loop
        x        := signed(data_i((i+1)*8-1 downto i*8));
        temp_acc := temp_acc + resize(x * WEIGHTS(j, i), 32);
    end loop;
    acc(j) <= temp_acc;
end loop;

-- Registered output: right-shift requantisation + saturation clipping
if rising_edge(clk) then
    shifted := shift_right(acc(j), SHIFT);
    -- clamp to [-128, 127] ...
end if;
```

Latency is 1 clock cycle per linear layer (ReLU is zero-cycle, fully combinational).

## Extending the library

Adding a new operator requires three steps:

**1.** Subclass `VHDLOperator` in `digitalify/operators/`:

```python
from digitalify.operators.base import VHDLOperator
from digitalify.ir.node import IRNode

class Conv2dOperator(VHDLOperator):
    def component_declaration(self, node: IRNode) -> str: ...
    def component_architecture(self, node: IRNode) -> str: ...
    def instantiation(self, node, sig_in, sig_out, sig_vi, sig_vo) -> str: ...
```

**2.** Register it in `digitalify/operators/registry.py`:

```python
from digitalify.operators.conv import Conv2dOperator
OPERATOR_REGISTRY["conv2d"] = Conv2dOperator
```

**3.** Add recognition logic in `digitalify/frontend/torch_importer.py` so that `nn.Conv2d` nodes are mapped to `op_type="conv2d"` and their weight tensor is stored in `IRNode.attrs`.

## Design decisions

**Why torch.fx and not ONNX?**
Reading directly from the `torch.fx` graph preserves all quantization metadata (per-layer scale and zero-point) without protobuf parsing or opset-version uncertainty. It also gives direct access to live Python module objects, making weight extraction straightforward. ONNX support is planned as a second frontend — the `IRGraph` intermediate representation is the natural seam for plugging it in, since the VHDL backend would not need any changes.

**Why INT8 fixed-point?**
INT8 is the dominant quantization format for on-device inference and maps cleanly to FPGA DSP slices. The accumulator is kept at 32 bits to avoid overflow for typical layer sizes (up to ~16 M input elements). The requantization step (right-shift + saturate) is a standard technique that synthesises efficiently on both FPGAs and ASICs.

**Why hardcoded weight constants?**
Embedding weights as VHDL `constant` arrays means the design is fully self-contained: no external memory bus, no DMA, no boot sequence. The synthesiser infers them as distributed LUTs or block-RAM depending on size, giving predictable timing. For very large models, a future option could target external memory interfaces instead.

## Project structure

```
digitalify/
├── __init__.py               # top-level convert() entry point
├── exceptions.py             # UnsupportedOperatorError, VHDLGenerationError
├── ir/
│   ├── node.py               # IRNode dataclass
│   └── graph.py              # IRGraph (ordered node list + shapes)
├── frontend/
│   └── torch_importer.py     # torch.fx GraphModule → IRGraph
├── operators/
│   ├── base.py               # abstract VHDLOperator
│   ├── registry.py           # op_type → handler class mapping
│   ├── linear.py             # LinearOperator
│   └── activations.py        # ReLUOperator
└── backend/
    └── vhdl/
        └── generator.py      # IRGraph → dict[filename, VHDL source]

vivado_api/
├── main.py                   # FastAPI application and all routes
├── config.py                 # settings (env vars) and board catalog
├── models.py                 # Pydantic request / response schemas
├── job_store.py              # file-based job persistence
├── vivado/
│   ├── runner.py             # TCL generation + Vivado subprocess wrapper
│   └── parsers.py            # Vivado report text → structured data
├── requirements.txt
└── .env.example              # environment variable template

examples/
└── simple_network.ipynb      # end-to-end walkthrough notebook

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
| `VIVADO_API_VIVADO_PATH` | `/tools/Xilinx/Vivado/2024.1/bin/vivado` | Path to the Vivado binary |
| `VIVADO_API_WORKSPACE_DIR` | `/tmp/vivado_api` | Root directory for uploads and job artefacts |
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

### Typical workflow

```
1.  GET  /boards                       → pick a board_id

2.  POST /upload  (multipart files)    → get file_id tokens for each .vhd

3.  POST /jobs    {board_id, top_entity, file_ids}
                                       → get job_id  (status: pending)

4.  POST /jobs/{id}/synthesize         → trigger Vivado  (status: synthesizing)
    GET  /jobs/{id}                    → poll until synthesized or failed

5.  GET  /jobs/{id}/synthesis          → utilization, timing, power + log tail
    GET  /jobs/{id}/schematic          → download schematic PDF
    GET  /jobs/{id}/reports            → full parsed reports + raw text

6.  POST /upload  (testbench file)     → get testbench file_id
    POST /jobs/{id}/simulate  {testbench_file_id, testbench_top, runtime}
                                       → trigger xsim  (status: simulating)
    GET  /jobs/{id}                    → poll until done or failed
    GET  /jobs/{id}/simulation         → log tail and elapsed time
```

### Reports returned

After synthesis completes, three structured reports are available:

**Utilization** — LUT, flip-flop, DSP, BRAM counts with percentage of device used.

**Timing** — worst-case setup slack (WNS), total negative slack (TNS), hold slack (WHS), and a boolean `timing_met` field.

**Power** — total on-chip power, split into dynamic and static components, plus junction temperature and confidence level.

### End-to-end with digitalify

```python
import digitalify, requests

# Generate VHDL locally
files = digitalify.convert(gm, input_shape=(16,))

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
    "top_entity": "network_top",
    "file_ids": file_ids,
}).json()

requests.post(f"{BASE}/jobs/{job['job_id']}/synthesize", headers=HDR)

# Poll until done, then fetch reports
import time
while True:
    status = requests.get(f"{BASE}/jobs/{job['job_id']}", headers=HDR).json()["status"]
    if status in ("synthesized", "failed"):
        break
    time.sleep(5)

reports = requests.get(f"{BASE}/jobs/{job['job_id']}/reports", headers=HDR).json()
print("LUTs used :", reports["utilization"]["luts"])
print("Timing met:", reports["timing"]["timing_met"])
print("Power (W) :", reports["power"]["total_power_w"])
```

---

## License

MIT
