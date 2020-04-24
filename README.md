# conllutils

**CoNLL-Utils** is a Python library for processing of [CoNLL-U](https://universaldependencies.org) treebanks. It
provides mutable Python types for the representation of tokens, sentences and dependency trees. Additionally, the 
sentences can be indexed into the compact numerical representation with data stored in the [NumPy](https://numpy.org)
arrays, that can be directly used as the instances for the _machine learning_ algorithms.

The library also provides a flexible _pipeline_ API for the treebank pre-processing, which allows you to:
* parse and write data from/to CoNLL-U files,
* filter or transform sentences, tokens or token's fields using the arbitrary Python function,
* filter only the lexical words (i.e. without empty or multiword tokens),
* filter only sentences which can be represented as the (non)projective dependency trees,
* extract only [Universal dependency](https://universaldependencies.org/u/dep/index.html) relations without 
  the language-specific extensions for DEPREL and DEPS fields,
* generate concatenated UPOS and FEATS field,
* extract the text of the sentences reconstructed from the tokens,
* replace the field's values matched by the regular expressions, or replace the missing values,
* create unlimited data stream, randomly shuffle data and form batches of instances
* ...
and more.

### Installation

The CoNLL-Utils is available on [PyPi](https://pypi.python.org/pypi) and can be installed via `pip`. To install simply
run:
```
pip install conllutils
```
The library supports Python 3.6 and later.

