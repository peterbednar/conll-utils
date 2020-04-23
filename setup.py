from setuptools import setup
from os import path

VERSION = "1.1.0"

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="conllutils",
    packages=["conllutils"],
    version=VERSION,
    license="MIT",
    description="Utility classes and functions for parsing and indexing files in CoNLL-U format.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=u"Peter Bednár",
    author_email="peter.bednar@tuke.sk",
    url="https://github.com/peterbednar/conllutils",
    install_requires=["numpy"],
    python_requires='>=3.6'
)