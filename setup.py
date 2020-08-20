from setuptools import setup

setup(
    name="pyronan",
    version="0.1",
    description="Framework and utilities for training models with Pytorch",
    url="http://github.com/rronan/pyronan",
    author="rronan",
    license="MIT",
    packages=["pyronan"],
    zip_safe=False,
    install_requires=["torch==1.5.0", "numpy"],
)
