from setuptools import find_packages, setup

setup(
    name="digitalify",
    version="0.2.0",
    description="Convert quantized QONNX models to bit-exact synthesisable VHDL",
    author="Riccardo Berta",
    author_email="riccardo.berta@unige.it",
    packages=find_packages(include=["digitalify", "digitalify.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "onnx>=1.15",
        "qonnx>=1.0",
    ],
    extras_require={
        "dev": ["pytest>=8", "jupyter", "matplotlib"],
        "examples": ["torch", "brevitas", "onnxoptimizer"],
    },
)
