# import os

from setuptools import setup

setup(
    name="multiplexer",
    version="1.0",
    description=("Multiplexed OCR"),
    url="https://github.com/fairinternal/MultiplexedOCR",
    packages=["multiplexer", "virtual_fs"],
    install_requires=[
        "yacs>=0.1.8",
    ],
)
