from setuptools import setup
from os import path

VERSION = "1.1.3"

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="conllutils",
    packages=["conllutils"],
    version=VERSION,
    license="MIT",
    description="A library for processing of CoNLL-U treebanks.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=u"Peter Bednár",
    author_email="peter.bednar@tuke.sk",
    url="https://github.com/peterbednar/conllutils",
    install_requires=["numpy"],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Utilities'
    ]
)