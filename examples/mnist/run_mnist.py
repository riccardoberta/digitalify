"""MNIST CNN end to end: Brevitas QONNX -> integer model -> VHDL -> GHDL.

    python examples/mnist/train_brevitas.py      # once: trains and exports mnist_cnn.onnx
    python examples/mnist/run_mnist.py [--sim 10] [--out vhdl_out/mnist]

1. imports the QONNX model and prints the hardware stages;
2. measures the accuracy of the bit-exact integer model on the 10 000 test
   images (it is the accuracy of the generated hardware);
3. checks the integer model against QONNX on real test images;
4. simulates the VHDL in GHDL on real test images, with random backpressure;
5. writes the VHDL.
"""

import argparse
import os
import warnings

import numpy as np

import digitalify
from digitalify.testing.reference import run_reference

from train_brevitas import HERE, load_mnist

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(HERE, "mnist_cnn.onnx"))
    ap.add_argument("--qonnx", type=int, default=200, help="images checked against QONNX")
    ap.add_argument("--sim", type=int, default=10, help="images simulated in GHDL (0: skip)")
    ap.add_argument("--out", default="vhdl_out/mnist", help="VHDL output directory")
    args = ap.parse_args()

    ir = digitalify.from_qonnx(args.model)
    print(ir.summary(), "\n")

    _, _, images, labels = load_mnist()
    # The network was trained on pixels / 255: quantise them as its input Quant does.
    x = ir.quantize_input(images.astype(np.float32)[:, None] / np.float32(255.0))

    pred = run_reference(ir, x).argmax(axis=1)
    print(f"integer model accuracy: {(pred == labels).mean():.4f} on {len(labels)} test images")

    check = digitalify.verify(args.model, ir=ir, inputs=x[:args.qonnx])
    print("integer model vs QONNX:", check.summary())

    if args.sim:
        sim = digitalify.simulate(ir, inputs=x[:args.sim])
        print("VHDL vs integer model: ", sim.summary())
        fast = digitalify.simulate(ir, inputs=x[:args.sim], backpressure=False)
        per_image = (fast.cycles - fast.first_output) / max(1, args.sim - 1)
        print(f"throughput: {per_image:.0f} clocks per image, "
              f"latency {fast.first_output} clocks")

    files = digitalify.convert(ir, output_dir=args.out)
    print(f"wrote {len(files)} VHDL files to {args.out}")


if __name__ == "__main__":
    main()
