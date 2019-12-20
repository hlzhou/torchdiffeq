import setuptools

setuptools.setup(
    name="njsde",
    version="0.0.1",
    author="Ricky Tian Qi Chen",
    author_email="rtqichen@cs.toronto.edu",
    description="ODE solvers and adjoint sensitivity analysis in PyTorch.",
    url="https://github.com/rtqichen/torchdiffeq",
    packages=['njsde', 'njsde._impl'],
    install_requires=['torch>=0.4.1'],
    classifiers=(
        "Programming Language :: Python :: 3"),)
