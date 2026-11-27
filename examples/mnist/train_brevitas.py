"""Train a small quantised CNN on MNIST with Brevitas and export it to QONNX.

    python examples/mnist/train_brevitas.py [--epochs 3] [--out examples/mnist/mnist_cnn.onnx]

The MNIST files are downloaded once into ``examples/mnist/data`` (git-ignored).

Network (4-bit weights and activations, 8-bit input and output):

    QuantIdentity(8b, unsigned)          input pixels, x / 255 quantised to 8 bits
    QuantConv2d(1 -> 8, 3x3)  + BN + QuantReLU(4b) + MaxPool 2   28 -> 26 -> 13
    QuantConv2d(8 -> 16, 3x3) + BN + QuantReLU(4b) + MaxPool 2   13 -> 11 -> 5
    Flatten + QuantLinear(400 -> 10) + QuantIdentity(8b)
"""

import argparse
import gzip
import os
import struct
import sys
import urllib.request

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.export import export_qonnx
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def load_mnist(data_dir=os.path.join(HERE, "data")):
    """Return (train_x, train_y, test_x, test_y): uint8 images (n, 28, 28) and labels."""
    os.makedirs(data_dir, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(MIRROR + name, path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if key.endswith("_x"):
            n, rows, cols = struct.unpack(">III", raw[4:16])
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(n, rows, cols)
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=8)
    return out["train_x"], out["train_y"], out["test_x"], out["test_y"]


class MnistCNN(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.inp = qnn.QuantIdentity(act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(1, 8, 3, weight_bit_width=bits, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = qnn.QuantReLU(bit_width=bits, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = qnn.QuantConv2d(8, 16, 3, weight_bit_width=bits, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = qnn.QuantReLU(bit_width=bits, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = qnn.QuantLinear(16 * 5 * 5, 10, weight_bit_width=bits, bias=True)
        self.out = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat)

    def forward(self, x):
        x = self.inp(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        return self.out(self.fc(x))


def to_tensor(images):
    return torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)


def accuracy(model, x, y, batch=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            pred = model(to_tensor(x[i:i + batch])).argmax(1).numpy()
            correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def train(epochs=3, batch=128, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y, test_x, test_y = load_mnist()
    model = MnistCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(train_x) // batch))
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(train_x))
        for i in range(0, len(order) - batch + 1, batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(to_tensor(train_x[idx])),
                                   torch.from_numpy(train_y[idx].astype(np.int64)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}, "
              f"test accuracy {accuracy(model, test_x, test_y):.4f}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "mnist_cnn.onnx"))
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):   # the torch exporter prints emoji
        stream.reconfigure(encoding="utf-8", errors="replace")
    model = train(args.epochs).eval()
    export_qonnx(model, torch.zeros(1, 1, 28, 28), export_path=args.out)
    # torch.onnx may store the weights in a side file: keep a single .onnx file.
    onnx.save(onnx.load(args.out), args.out)
    if os.path.exists(args.out + ".data"):
        os.remove(args.out + ".data")
    print("exported", args.out)


if __name__ == "__main__":
    main()
