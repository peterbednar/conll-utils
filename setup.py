import os
from setuptools import setup

VERSION = "0.9.0"

setup(
    name="CoNLLUtils",
    packages=["conllutils"],
    version=VERSION,
    license="MIT",
    description="Utility classes and functions for parsing and indexing of files in CoNLL-U format.",
    author=u"Peter Bednár",
    author_email="peter.bednar@tuke.sk",
    url="https://github.com/peterbednar/conllutils",
    install_requires=["numpy"]
)