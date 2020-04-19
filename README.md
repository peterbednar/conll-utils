# conllutils

Utility classes and functions for parsing and indexing files in CoNLL-U format.

## Code samples

### Working with sentences and tokens

```python
from conllutils import Sentence, Token, FORM

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

# sentences and tokens are list and dictionary types, use indexing to access words and fields
first = sentence[0]
print(first['form'])        # field keys are in lower-case
print(first[FORM])          # library defines constants for field names
print(first.upos)           # fields are accessible also as the token attributes
print(first.feats['Case'])  # FEATS parsed to dictionaries

# you can modify tokens and sentences or create a new one
dot = Token(id=5, form='.', lemma='.', upos='PUNCT', head=2, deprel='punct')
sentence.append(dot)        # add '.' at the end of the sentence
print(sentence.to_conllu()) # print modified sentence in CoNLL-U format
print(sentence.text)        # print raw text reconstructed from tokens

# transform sentence to dependency tree representation
tree = sentence.to_tree()
print(tree.root.token.form) # print root FORM
print(tree.is_projective()) # is tree projective?
for child in tree.root:
    print(child.token.form) # iterate over root's direct children
for leaf in tree.leaves():
    print(leaf.token.form)  # iterate over all leaves
```
