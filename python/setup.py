import os
from setuptools import setup

VERSION = "0.9.0"

setup(
    name="CoNLLUtils",
    packages=["conllutils"],
    version=VERSION,
    long_description=open(os.path.join(os.path.dirname(__file__), "../README.md")).read(),
    long_description_content_type="text/markdown",
    author=u"Peter Bednár",
    author_email="peter.bednar@tuke.sk",
    url="https://github.com/peterbednar/conllutils",
)