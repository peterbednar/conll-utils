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
* extract only [Universal dependency relations](https://universaldependencies.org/u/dep/index.html) without 
  the language-specific extensions for DEPREL and DEPS fields,
* generate concatenated UPOS and FEATS field,
* extract the text of the sentences reconstructed from the tokens,
* replace the field's values matched by the regular expressions, or replace the missing values,
* create unlimited data stream, randomly shuffle data and form batches of instances
* ... and more.

### Installation

The library supports Python 3.6 and later.

#### pip

The CoNLL-Utils is available on [PyPi](https://pypi.python.org/pypi) and can be installed via `pip`. To install simply
run:
```
pip install conllutils
```

To upgrade the previous installation to the newest release, use:
```
pip install conllutils -U
```

#### From source

Alternatively, you can also install library from this git repository, which will give you more flexibility and allows
you to start contributing to the CoNLL-Utils code. For this option, run:
```
git clone https://github.com/peterbednar/conllutils.git
cd conllutils
pip install -e .
```

### Getting started with CoNLL-Utils

#### Preparing pipeline for sentence pre-preprocessing

```python
from conllutils.pipeline import pipe

NUM_REGEX = r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+'
NUM_FORM = '__number__'

p = pipe()
p.only_words()
p.only_universal_deprel()
p.upos_feats()
p.lowercase('form')
p.replace('form', NUM_REGEX, NUM_FORM)
```

#### Indexing sentences for machine learning

```python
train_file = 'en_ewt-ud-train.conllu'
indexed_fields = {'form', 'upos_feats', 'deprel'}

index = pipe().read_conllu(train_file).pipe(p).create_index(fields=indexed_fields)
train_data = pipe().read_conllu(train_file).pipe(p).to_instance(index).collect()
```

#### Iterating over batches of training instances

```python
total_size = 10000
batch_size = 100

for batch in pipe(train_data).stream(total_size).shuffle().batch(batch_size):
    # update your model for the next batch of instances
    pass
```

### License

CoNLL-Utils is released under the MIT License. See the [LICENSE](https://github.com/peterbednar/conllutils/blob/master/LICENSE)
file for more details.
