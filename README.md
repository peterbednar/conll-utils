# conllutils

Utility classes and functions for parsing and indexing files in CoNLL-U format.

## Code samples

### Working with sentences and tokens

```python
from conllutils import Sentence, Token
from conllutils import FORM

s = """\
# sent_id = 2
# text = I have no clue.
1	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1	2	nsubj	_	_
2	have	have	VERB	VBP	Number=Sing|Person=1|Tense=Pres	0	root	_	_
3	no	no	DET	DT	PronType=Neg	4	det	_	_
4	clue	clue	NOUN	NN	Number=Sing	2	obj	_	SpaceAfter=No

"""
# parse Sentence object from a string
# by default, the values of FEATS field are stored as the strings
# to access Universal features directly, set parse_feats=True to parse FEATS values to dictionaries
sentence = Sentence.from_conllu(s, parse_feats=True)

# sentences and tokens are parsed as the list and dictionary types
# use indexing to access words and fields
first = sentence[0]
print(first['form'])    # field keys are in lower-case
print(first[FORM])      # library defines constants for field names
print(first.upos)       # fields are accessible also as the token attributes
print(first.feats['Case'])  # FEATS parsed to dictionaries

# you can modify tokens and sentences or create a new one
dot = Token(id=5, form='.', lemma='.', upos='PUNCT', head=2, deprel='punct')
sentence.append(dot)    # add '.' at the end of the sentence
# print modified sentence in CoNLL-U format
print(sentence.to_conllu())

# transform sentence to dependency tree representation
tree = sentence.to_tree()
print(tree.root.token.form) # print root FORM
print(tree.is_projective()) # is tree projective?
for child in tree.root:
    print(child.token.form) # iterate over root's direct children
for leaf in tree.leaves():
    print(leaf.token.form)  # iterate over all leaves
```

### Indexing for machine learning

```python
import numpy as np
from conllutils import FORM, LEMMA
from conllutils import read_conllu, create_index, map_to_instances, shuffled_stream
from conllutils import create_inverse_index

train_file = 'en_ewt-ud-train.conllu'
test_file = 'en_ewt-ud-test.conllu'

# the first pass over the training data - create an index

# read CoNLL-U file and return an interator over the parsed sentences
# skip empty and multiword tokens
train_sentences = read_conllu(train_file, skip_empty=True, skip_multiword=True)
# create an index of string values for all fields
# for FORM and LEMMA keep only the words with frequency of occurences >= 5
index = create_index(train_sentences, min_frequency={FORM: 5, LEMMA:5})

# print number of indexed values for each field
for field in sorted(index):
    # index[field] is a nested dictionary mapping string value to integer index
    print(f"{field}:\t{len(index[field])}")

# the second pass - map sentences to indexed instances

train_sentences = read_conllu(train_file, skip_empty=True, skip_multiword=True)
train_instances = map_to_instances(train_sentences, index)
# train_instances is an iterator over training instances

# model training

np.random.seed(1)   # init random state
# iterate over randomly shuffled training data and form the batches
for batch in shuffled_stream(train_instances, total_size=10000, batch_size=100):
    # each batch is a list of 100 instances/sentences
    for instance in batch:
        # length of the instance (number of words in the sentence)
        num_words = instance.length
        # instance values are NumPy np.int arrays of continuous indexes for the whole sentence
        forms = instance.form
        heads = instance.head
        # here you can update the model parameters according to instance values
        # ...

# model testing

# iterate over the testing data
test_sentences = read_conllu(test_file, skip_empty=True, skip_multiword=True)
# for the indexing use the index generated from the training sentences
test_instances = map_to_instances(test_sentences, index)
for instance in test_instances:
    # here you can call your model to predict outputs for a testing instance
    pass

# scoring of the new sentence

s = """\
# sent_id = 1
# text = They buy and sell books.
1	They	they	PRON	PRP	Case=Nom|Number=Plur	_	_	_	_
2	buy	buy	VERB	VBP	Number=Plur|Person=3|Tense=Pres	_	_	_	_
3	and	and	CCONJ	CC	_	_	_	_	_
4	sell	sell	VERB	VBP	Number=Plur|Person=3|Tense=Pres	_	_	_	_
5	books	book	NOUN	NNS	Number=Plur	_	_	_	SpaceAfter=No
6	.	.	PUNCT	.	_	_	_	_	_

"""
# index instance
sentence = Sentence.from_conllu(s)
instance = sentence.to_instance(index)

# here you can call your model to predict outputs for a new instance
# add HEAD and DEPREL
instance.head = np.array([2, 0, 4, 2, 2, 2], dtype=np.int)
# [3, 5, 14, 11, 7, 1] are corresponding indexes of "nsubj", "root", "cc", "conj", "obj", "punct"
instance.deprel = np.array([3, 5, 14, 11, 7, 1], dtype=np.int)

# revert instance to sentence
inverse_index = create_inverse_index(index)
scored = instance.to_sentence(inverse_index)
# print sentence with the new fields
print(scored.to_conllu())
```