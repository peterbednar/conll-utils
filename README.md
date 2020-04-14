# CoNLLUtils

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
# to access Universal features directly, use parse_feats=True option to parse FEATS values to dictionaries
sentence = Sentence.from_conllu(s, parse_feats=True)

# sentences are parsed as the lists and tokens as the dictionaries, use indexing to access words and fields
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
```

