# Dataflow architecture

Design for moving digitalify from "one vector in, one vector out" MLPs to a
general streaming dataflow architecture that supports CNNs and other layer
types, on any FPGA vendor.

Status: (1) streaming core, (2) Conv / pooling and (3) common QONNX layers are
implemented and verified (bit-exact in GHDL with random backpressure, and
synthesised with Vivado).  Residual graphs and an AXI4-Stream wrapper come later.

## Goals

* Support convolutional networks and the layer types common in QONNX
  exports (Brevitas, QKeras, FINN), not just MLPs.
* Stay vendor-neutral: plain VHDL-2008, no vendor primitives or IP.
* Keep bit-exactness: every generated design is checked against the integer
  reference model in simulation, and the reference model against QONNX.
* Let the user trade area for throughput per layer.

Non-goals for now: residual / branching graphs (needs FIFOs to balance
branches), memory-mapped weights (weights stay constants), multi-clock designs.

## 1. Streams

Every tensor travels as a **stream of beats**.

* A tensor has a stream shape `(H, W, C)`.  One beat carries one pixel: its
  `C` elements, packed as today (element `c` at `data((c+1)*B-1 downto c*B)`).
  Beats come in raster order (row by row, left to right).
* A plain vector is the special case `(1, 1, N)`: one beat per vector, which
  is exactly the current MLP interface.
* ONNX tensors are NCHW.  The importer maps a 3-D activation `(C, H, W)` to the
  stream `(H, W, C)`.  It permutes the weights of a `Flatten → MatMul`, so the
  hardware can flatten in HWC order while the maths stays identical.
* A network input of rank 3 is streamed as an image only when it feeds a
  convolution or pooling layer; otherwise it is flattened to one vector beat.

The bus width is `C × bits`, fixed per tensor.  Within one beat, channel
parallelism is total; folding (section 4) happens inside the compute units,
not on the bus.

## 2. Handshake

Every stage, and the top level, has the same ports:

```
clk, rst
in_valid,  in_ready,  in_data      -- upstream
out_valid, out_ready, out_data     -- downstream
```

A beat moves when `valid = '1'` and `ready = '1'` on the same rising edge,
with AXI4-Stream rules: a source never waits for `ready` before asserting
`valid`, and once `valid` is asserted, it and the data are held until the
transfer.  The same handshake maps directly onto AXI4-Stream (`tvalid`/`tready`/
`tdata`) and Intel Avalon-ST, so vendor wrappers stay thin.

### Stage structure: core + skid buffer

```
            ce                     ┌───────────────┐
in ──► [ stage core ] core_valid ─►│  skid buffer  │──► out
            ▲          core_data   │ (2 registers) │
            └────────── ce ◄───────┤ in_ready (reg)│◄── out_ready
                                   └───────────────┘
```

* The **core** holds all the logic and registers, and advances only when
  `ce = '1'`.  With `ce = '0'` it freezes completely.  Its output register
  (`core_valid`, `core_data`) is consumed on every `ce = '1'` edge.
* **`ce` is the skid buffer's `in_ready`**, which is a register output.
  So no stage has a combinational path from `out_ready` to `in_ready`: the ready
  signal never ripples through the network, and the timing of a deep network
  does not degrade.
* Cores accept input with `in_ready = ce and <core can accept>`, where the
  second term comes from the core's own state registers.
* Cost: two output registers per stage.

Pipelines inside a core (the `pipeline` option of `linear`) use the same `ce`,
so they keep working unchanged, now stallable.

## 3. Hardware stages

Each ONNX pattern is lowered into a chain of small stages.  Composition keeps
every stage simple and testable on its own.

| Stage | Function | Rate |
|---|---|---|
| `linear` | `y = requant(relu(W·x + b))` per beat (matrix-vector unit) | 1 beat/clock when fully parallel; folded: 1 beat per `NF·SF` clocks |
| `relu` | `max(0, x)` per element | 1/clock |
| `pad` | inserts border pixels with a constant value | emits `Hp·Wp` beats per `H·W` input beats |
| `swg` | sliding window generator: line buffers + window registers; emits a `KH·KW·C` window beat per output position (stride applied) | 1 input beat/clock |
| `pool` | per-channel max or sum over a window beat (balanced tree, optional `"pool"` register halfway) | 1/clock |
| `global_pool` | per-channel sum over all `H·W` beats → one beat | 1 input beat/clock |
| `flatten` | collects `H·W` beats into one `(1, 1, H·W·C)` beat | 1 input beat/clock |
| `lut` | elementwise table lookup (any activation followed by `Quant`) | 1/clock |
| `requant` | elementwise `clip(round(x·m / 2^n))` (a standalone `Quant`) | 1/clock |

ONNX lowering:

* **Conv** (any kernel, stride, padding, groups): `[pad] → swg → linear`.  The
  window beat order is `(kh, kw, c)`, and the weights are reshaped to
  `(C_out, KH·KW·C_in)` in that order.  Grouped and depthwise convolutions use a
  block-sparse weight matrix: zero weights are constant-folded away in fully
  parallel mode.  A 1×1 convolution with stride 1 and no padding is a bare
  `linear` on the pixel stream.
* **MaxPool / AveragePool**: `[pad] → swg → pool`.  MaxPool pads with the
  smallest value of the input type.  AveragePool becomes a window sum whose
  scale is divided by `KH·KW` (padding is not supported, since
  `count_include_pad` would change the divisor per position).
* **GlobalAveragePool**: `global_pool`, with scale divided by `H·W`.
* **BatchNormalization** and per-channel `Mul` / `Add` by constants directly
  after a Conv / MatMul / Gemm are folded into the `linear` stage: they change
  the accumulator scale and the bias, and a negative factor flips the sign of
  that channel's weights.  The existing fractional-bias mechanism keeps
  non-integer biases exact up to `F` bits.
* **Activations other than ReLU** (Sigmoid, Tanh, Clip/HardTanh, LeakyRelu,
  HardSigmoid, Elu, ...), and any chain of elementwise ops ending in a `Quant`,
  become a `lut`.  The table is computed by running that ONNX sub-graph with
  qonnx on every possible input value, so it matches QONNX by construction.
  This requires a small input range (at most 4096 values per channel).
* **Threshold activations.** For outputs with few bits, the `linear` stage can
  replace bias + ReLU + `Quant` by `2^B − 1` sorted integer thresholds per
  channel on the raw sum (FINN's `MultiThreshold`):
  `y = BASE[j] + #{k : acc >= T[j][k]}`.
  * The quantiser is monotone in the sum, so the thresholds are computed
    exactly with rational arithmetic, for every rounding mode.  The result is
    exact for non-dyadic scales and folded BatchNorm, where the multiplier form
    is only approximate.
  * Thresholds that every reachable sum passes are folded into `BASE`.
  * The comparisons form a thermometer code, counted by binary search
    (`log2` multiplexer levels).
  * `thresholds="auto"` uses them when the multiplier form is inexact and the
    output has at most 4 bits.
  * On the MNIST example the accumulators shrink from 31 to 15 and from 28 to
    12 bits, and the 32-bit requantisation multipliers disappear.
* **Quant** on an already quantised tensor becomes a `requant` stage.
* **Flatten** of a spatial tensor becomes a `flatten` stage, plus the column
  permutation of the following weights.  Of a `(1, 1, C)` tensor, it is a no-op.

## 4. Folding (area / throughput)

`linear` takes two parallelism parameters per layer:

* `PE`: output neurons computed in parallel (divides `out_features`);
* `SIMD`: inputs multiplied in parallel per neuron (divides `in_features`).

`PE = out`, `SIMD = in` is the current fully parallel unit (one beat per clock,
with the `pipeline` options).  Otherwise the unit is folded:

* It needs `PE·SIMD` multipliers and takes `NF·SF` clocks per beat, where
  `NF = out/PE` and `SF = in/SIMD`.
* The input vector is latched and rotated by `SIMD` elements per clock.  The
  next beat is latched while the last work item of the current one issues, so
  there is no bubble between beats.
* The weights live in a constant ROM indexed by the fold counters.  Synthesis
  maps it to LUTs or block RAM.
* `PE` accumulators sum over `SF` clocks.  A finalisation register then adds
  the bias, applies ReLU and requantises `PE` neurons per clock into the output
  vector, and the output beat goes out after the last neuron group.
* The datapath is a pipeline of work items (`PE` neurons × `SIMD` inputs).
  Each item carries tags through the registers: valid, first / last synapse
  fold, neuron fold.  The `pipeline` option places registers as in the fully
  parallel unit: weight-ROM word and products (`products`), SIMD adder tree
  (`tree`), after bias (`acc`), after the requantisation product or threshold
  comparisons (`mult`), after the rounding decision (`round`).  On the MNIST
  example this takes the folded design from 18 to 174 MHz.
* Rounding pipeline (`round`, both architectures).  `clip(floor + inc)` is
  decided from the floor alone: `floor >= CLIP_MAX` gives `CLIP_MAX`,
  `floor < CLIP_MIN` gives `CLIP_MIN`, otherwise `floor + inc` fits the output.
  So the first clock registers the low `Y_W` bits of the floor, the round-up
  bit and two saturation flags.  The second clock is a `Y_W`-bit increment and
  a multiplexer.
* No wide comparison in the rounding.  The discarded bits `r` are compared
  with `half = 2^(SHIFT-1)` through the guard bit `p(SHIFT-1)` and the OR of
  the sticky bits below it (`r > half` = guard and sticky, `r = half` = guard
  and not sticky, `r /= 0` = guard or sticky).  The saturation flags compare
  the floor on its `FL_W = P_W - min(SHIFT)` significant bits.  On the MNIST
  example the 64-bit carry chains go away: 166 → 174 MHz.
* Wide products (`mult:K`).  A product that does not fit one 25 × 18 DSP
  multiplier becomes a cascade of DSP blocks.  K registers after it let
  synthesis fill each block's internal registers.  `pipeline=True` uses K = 3
  for these products, which brings MNIST to 176 MHz (parallel) and 179 MHz
  (folded).  4 registers give nothing more.
* Pool tree.  `pool` reduces each channel's window with a balanced max /
  sum tree (`ceil(log2 K)` levels instead of a chain of `K - 1`).  The
  `"pool"` pipeline stage registers it halfway when the tree has 2 or more
  levels.  The sums wrap at the tree width like the chain did, so results are
  unchanged.  On the folded MNIST this removes the last path over 5 ns:
  179 → 214 MHz.

The user passes `folding={"linear_0": (PE, SIMD), ...}` to `convert` /
`simulate`.  Layers not listed stay fully parallel.  A helper that chooses
`PE`/`SIMD` for a target throughput can come later; the interface does not
depend on it.

For a convolution, one output pixel is one `linear` beat, so the frame rate is
`Ho·Wo / (beats per clock of the slowest stage) / f_clk`.  Folding the `linear`
of a conv layer trades multipliers against frame rate in exactly that ratio.

## 5. Bit-exactness and verification

* **Reference model** (`digitalify.testing.reference`): operates on
  `(n, H, W, C)` integer tensors, one function per stage.  It is the
  specification of the hardware.
* **`verify`**: converts stream-order integers to NCHW, runs QONNX and maps
  the outputs back.  Float-only steps (BatchNorm folding, non-dyadic scales,
  average divisors) can differ by one LSB at rounding ties.  These are reported
  as approximate, as the requantisation is today.
* **`simulate`**: a new streaming testbench.
  * It sends all beats of `n` test tensors back to back.  With
    `backpressure=True` (the default), it drives `in_valid` and `out_ready`
    from pseudo-random patterns.
  * It compares every output beat in order and counts a test tensor as passed
    when all of its beats match.
  * It checks the protocol: `out_valid` and `out_data` must stay stable while
    `out_ready = '0'`.
  * It fails on a timeout if beats are missing.
  * It reports cycle counts (first output, total), so throughput and latency
    are measured rather than computed.  Exact cycle-latency checks are dropped,
    because with stalls latency is not a constant any more.

## 6. Code layout

```
digitalify/
├── ir/                       IRNode shapes become stream shapes (H, W, C)
├── frontend/qonnx_importer   lowering of the ONNX patterns above
├── operators/
│   ├── base.py               ports, core/skid wrapping, pipeline option
│   ├── skid.py               shared skid-buffer entity (<net>_skid)
│   ├── linear.py             fully parallel + folded MVAU
│   ├── window.py             pad, swg, pool, global_pool, flatten
│   └── elementwise.py        relu, lut, requant
├── backend/vhdl/generator    top level wires valid/ready/data between stages
└── testing/                  reference model, streaming testbench, CNN builder
```

## 7. Later

* Residual `Add` / `Concat` (DAG graphs with balancing FIFOs).
* AXI4-Stream / Avalon-ST top-level wrappers.
* Block-RAM line buffers with synchronous read, for large images (the first
  version uses distributed RAM with asynchronous read).
* Automatic folding for a target throughput or resource budget.
* Bias adder into the DSP input: when a requantisation product follows the
  `acc` register directly (no ReLU), synthesis absorbs that register into the
  DSP and the last tree level plus the bias becomes the critical path (5 ns on
  the parallel MNIST, 176 MHz).  Registering the last tree level separately
  would fix it.
* Padding inside the window generator.  A separate `pad` stage emits
  `Hp·Wp` beats per `H·W` input pixels, so a padded 3×3 convolution on a small
  image runs at `H·W / (Hp·Wp)` of the input rate (64/100 on 8×8).
