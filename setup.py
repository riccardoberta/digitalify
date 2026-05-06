from setuptools import setup, find_packages

setup(
    name="digitalify",
    version="0.1.0",
    description="Convert quantized ML models to synthesisable VHDL",
    author="Riccardo Berta",
    author_email="riccardo.berta@unige.it",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": ["jupyter", "matplotlib"],
    },
)
