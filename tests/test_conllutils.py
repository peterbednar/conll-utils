import os
import pytest

from conllutils import *
from conllutils import _create_dictionary

class _StringIO(StringIO):

    def close(self):
        pass

    def release(self):
        super().close()

def _data_filename(name):
    return os.path.join(os.path.dirname(__file__), name)

def _read_file(name):
    with open(name, "rt", encoding="utf-8") as f:
        return f.read()

def _fields(*values):
    return {FIELDS[i] : v for i, v in enumerate(values) if v is not None}

_DATA1_CONLLU = """\
1-2	vámonos	_	_	_	_	_	_	_	_
1	vamos	ir	_	_	_	0	_	_	_
2	nos	nosotros	_	_	_	1	_	_	_
3-4	al	_	_	_	_	_	_	_	_
3	a	a	_	_	_	5	_	_	_
4	el	el	_	_	_	5	_	_	_
5	mar	mar	_	_	_	1	_	_	_

1	Sue	Sue	_	_	_	_	_	_	_
2	likes	like	_	_	_	_	_	_	_
3	coffee	coffee	_	_	_	_	_	_	_
4	and	and	_	_	_	_	_	_	_
5	Bill	Bill	_	_	_	_	_	_	_
5.1	likes	like	_	_	_	_	_	_	_
6	tea	tea	_	_	_	_	_	_	_

"""

@pytest.fixture
def data1():
    return _data_filename("data1.conllu")
@pytest.fixture
def data2():
    return _data_filename("data2.conllu")
@pytest.fixture
def data3():
    return _data_filename("data3.conllu")
@pytest.fixture
def data4():
    return _data_filename("data4.conllu")
@pytest.fixture
def data5():
    return _data_filename("data5.conllu")

def test_token():
    token = Token()
    assert str(token) == "<None,None,None>"
    token[ID] = multiword_id(1, 2)
    token[FORM] = "vámonos"
    assert str(token) == "<1-2,vámonos,None>"
    assert token.to_collu() == "1-2	vámonos	_	_	_	_	_	_	_	_"

    assert token.id == multiword_id(1, 2)
    token.id = 1
    assert token.id == 1 and token.form == "vámonos"
    del token.id
    assert ID not in token

    with pytest.raises(AttributeError):
        token.unknown
    with pytest.raises(AttributeError):
        del token.unknown
    token.unknown = 1
    assert token.unknown == 1

def test_dependency_tree(data1, data2):
    empty = Sentence().to_tree()
    assert len(empty) == 0
    assert list(empty.inorder()) == []

    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))
    tree0 = sentences[0].to_tree()
    assert tree0.root is not None
    assert str(tree0) == "<<1,vamos,None>,None,[<<2,nos,None>,None,[]>, <<5,mar,None>,None,[<<3,a,None>,None,[]>, <<4,el,None>,None,[]>]>]>"

    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))
    tree0 = sentences[0].to_tree()    
    assert tree0.root is not None
    assert str(tree0) == "<<2,buy,VERB>,root,[<<1,They,PRON>,nsubj,[]>, <<4,sell,VERB>,conj,[<<3,and,CONJ>,cc,[]>]>, <<5,books,NOUN>,obj,[]>, <<6,.,PUNCT>,punct,[]>]>"

    assert len(tree0) == len(sentences[0])
    assert [node.is_root for node in tree0] == [False, True, False, False, False, False]
    assert [node.is_leaf for node in tree0] == [True, False, True, False, True, True]
    assert [node.index for node in tree0] == list(range(len(tree0)))
    assert [node.token[FORM] for node in tree0] == ["They", "buy", "and", "sell", "books", "."]
    assert [len(node) for node in tree0] == [0, 4, 0, 1, 0, 0]

    root0 = tree0.root
    assert [root0[i] for i in range(len(root0))] == list(root0)
    assert root0[:] == list(root0)

    assert [node.token[FORM] for node in tree0.nodes] == ["They", "buy", "and", "sell", "books", "."]
    assert [node.token[FORM] for node in tree0.inorder()] == ["They", "buy", "and", "sell", "books", "."]
    assert [node.token[FORM] for node in tree0.preorder()] == ["buy", "They", "sell", "and", "books", "."]
    assert [node.token[FORM] for node in tree0.postorder()] == ["They", "and", "sell", "books", ".", "buy"]
    assert [leaf.token[FORM] for leaf in tree0.leaves()] == ["They", "and", "books", "."]

    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    instances = list(map_to_instances(sentences, index))

    tree0 = instances[0].to_tree()
    assert [node.token[FORM] for node in tree0] == [index[FORM][f] for f in ["They", "buy", "and", "sell", "books", "."]]

def test_read_conllu(data1):
    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))
    assert sentences == [[
            _fields(multiword_id(1, 2), "vámonos"),
            _fields(1, "vamos", "ir", None, None, None, 0),
            _fields(2, "nos", "nosotros", None, None, None, 1),
            _fields(multiword_id(3, 4), "al"),
            _fields(3, "a", "a", None, None, None, 5),
            _fields(4, "el", "el", None, None, None, 5),
            _fields(5, "mar", "mar", None, None, None, 1)
        ], [
            _fields(1, "Sue", "Sue"),
            _fields(2, "likes", "like"),
            _fields(3, "coffee", "coffee"),
            _fields(4, "and", "and"),
            _fields(5, "Bill", "Bill"),
            _fields(empty_id(5, 1), "likes", "like"),
            _fields(6, "tea", "tea"),
    ]]

def test_to_from_conllu(data2):
    sentences = list(read_conllu(data2))
    assert sentences[0].to_conllu() + "\n\n" + sentences[1].to_conllu() + "\n\n" == _read_file(data2)

    s = _read_file(data2)
    assert Sentence.from_conllu(s) == sentences[0]
    assert Sentence.from_conllu(s, multiple=True) == sentences

    with pytest.raises(ValueError):
        Sentence.from_conllu("# empty string")

    with pytest.raises(ValueError):
        Sentence.from_conllu("# empty string", multiple=True)
    
def test_parse_deps_feats(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False, parse_deps=True, parse_feats=True, upos_feats=False, normalize=None, split=None))
    assert sentences[0][0][FEATS] == {"Case":"Nom", "Number":"Plur"}
    assert sentences[0][0][DEPS] == {(2, "nsubj"), (4, "nsubj")}
    assert FEATS not in sentences[0][2]
    assert DEPS not in sentences[1][0]

    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False, parse_deps=False, parse_feats=False, upos_feats=False, normalize=None, split=None))
    assert sentences[0][0][FEATS] == "Case=Nom|Number=Plur"
    assert sentences[0][0][DEPS] == "2:nsubj|4:nsubj"
    assert FEATS not in sentences[0][2]
    assert DEPS not in sentences[1][0]

def test_upos_feats(data1, data2):
    sentences = list(read_conllu(data1, parse_deps=True, parse_feats=True, upos_feats=True, normalize=None, split=None))
    assert [UPOS_FEATS not in sentences[0][i] for i in range(len(sentences[0]))]

    sentences = list(read_conllu(data2, parse_deps=True, parse_feats=False, upos_feats=True, normalize=None, split=None))
    assert [sentences[0][i][UPOS_FEATS] for i in range(len(sentences[0]))] == [
            "POS=PRON|Case=Nom|Number=Plur",
            "POS=VERB|Number=Plur|Person=3|Tense=Pres",
            "POS=CONJ",
            "POS=VERB|Number=Plur|Person=3|Tense=Pres",
            "POS=NOUN|Number=Plur",
            "POS=PUNCT"
        ]
    
    sentences = list(read_conllu(data2, parse_deps=True, parse_feats=True, upos_feats=True, normalize=None, split=None))
    assert [sentences[0][i][UPOS_FEATS] for i in range(len(sentences[0]))] == [
            {"POS":"PRON","Case":"Nom","Number":"Plur"},
            {"POS":"VERB","Number":"Plur","Person":"3","Tense":"Pres"},
            {"POS":"CONJ"},
            {"POS":"VERB","Number":"Plur","Person":"3","Tense":"Pres"},
            {"POS":"NOUN","Number":"Plur"},
            {"POS":"PUNCT"}
        ]

def test_empty_multiword(data1):
    with pytest.raises(ValueError):
        empty_id(-1, 1)

    with pytest.raises(ValueError):
        empty_id(0, 0)

    with pytest.raises(ValueError):
        multiword_id(-1, 1)

    with pytest.raises(ValueError):
        multiword_id(0, 1)

    with pytest.raises(ValueError):
        multiword_id(1, 1)

    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False))

    assert [token.is_multiword for token in sentences[0]] == [True, False, False, True, False, False, False]
    assert [token.is_empty for token in sentences[0]] == [False, False, False, False, False, False, False]

    assert [token.is_multiword for token in sentences[1]] == [False, False, False, False, False, False, False]
    assert [token.is_empty for token in sentences[1]] == [False, False, False, False, False, True, False]

    assert [sentences[0].get(i)[FORM] for i in range(1, 6)] == ["vamos", "nos", "a", "el", "mar"]
    assert [sentences[1].get(i)[FORM] for i in range(1, 7)] == ["Sue", "likes", "coffee", "and", "Bill", "tea"]

    assert sentences[0].get(multiword_id(1,2))[FORM] == "vámonos"
    assert sentences[1].get(empty_id(5,1))[FORM] == "likes"
    assert sentences[0].get("1-2")[FORM] == "vámonos"
    assert sentences[1].get("5.1")[FORM] == "likes"

    with pytest.raises(IndexError):
        sentences[0].get(0)

    with pytest.raises(IndexError):
        sentences[0].get(6)

    with pytest.raises(IndexError):
        sentences[0].get(multiword_id(1,3))

    with pytest.raises(IndexError):
        sentences[0].get(empty_id(1,2))

    sentences = list(read_conllu(data1, skip_empty=True, skip_multiword=True))
    for sentence in sentences:
        for token in sentence:
            assert (not token.is_empty) and (not token.is_multiword)

def test_words_tokens(data1):
    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False))

    words0 = sentences[0].words()
    raw_tokens0 = sentences[0].raw_tokens()
    assert list(sentences[0].tokens()) == sentences[0]
    assert [token[FORM] for token in words0] == ["vamos", "nos", "a", "el", "mar"]
    assert [token[FORM] for token in raw_tokens0] == ["vámonos", "al", "mar"]

    words1 = sentences[1].words()
    raw_tokens1 = sentences[1].raw_tokens()
    assert list(sentences[1].tokens()) == sentences[1]
    assert [token[FORM] for token in words1] == ["Sue", "likes", "coffee", "and", "Bill", "tea"]
    assert [token[FORM] for token in raw_tokens1] == ["Sue", "likes", "coffee", "and", "Bill", "tea"]

def test_write_conllu(data1, data2, data3):
    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False, parse_deps=True, parse_feats=True, upos_feats=False, normalize=None, split=None))
    output = _StringIO()
    write_conllu(output, sentences)
    assert output.getvalue() == _DATA1_CONLLU
    output.release()

    input = _read_file(data2)
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False, parse_deps=True, parse_feats=True, upos_feats=False, normalize=None, split=None))
    output = _StringIO()
    write_conllu(output, sentences)
    assert output.getvalue() == input
    output.release()

    input = _read_file(data2)
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False, parse_deps=False, parse_feats=False, upos_feats=False, normalize=None, split=None))
    output = _StringIO()
    write_conllu(output, sentences)
    assert output.getvalue() == input
    output.release()

    input = _read_file(data3)
    sentences = list(read_conllu(data3, skip_empty=False, skip_multiword=False, parse_deps=True, parse_feats=True, upos_feats=False, normalize=None, split=None))
    output = _StringIO()
    write_conllu(output, sentences)
    assert output.getvalue() == input
    output.release()

def test_create_dictionary(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    dictionary = _create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD})

    assert dictionary.keys() == set(FIELDS)-{ID, HEAD}
    assert dictionary[FORM] == {"They":1, "buy":1, "and":1, "sell":1, "books":1, ".":2, "I":1, "have":1, "no":1, "clue":1}
    assert dictionary[FORM_NORM_CHARS] == {"e":4, "l":3, "o":3, "h":2, "y":2, "b":2, "u":2, "a":2, "n":2, "s":2, ".":2, "t":1, "d":1, "k":1, "i":1, "v":1, "c":1}

    with pytest.raises(ValueError):
        _create_dictionary(sentences, fields={ID})

    with pytest.raises(ValueError):
        _create_dictionary(sentences, fields={HEAD})

def test_create_index(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    dictionary = _create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD})

    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    assert [list(index[f].values()) for f in index.keys()] == [list(range(1, len(index[f])+1)) for f in index.keys()]
    assert [index[f].keys() for f in index.keys()] == [dictionary[f].keys() for f in index.keys()]

    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD}, min_frequency=2)
    assert index[FORM] == {".":1}
    assert index[FORM_NORM_CHARS] == {"e":1, "o":2, "l":3, "y":4, "u":5, "s":6, "n":7, "h":8, "b":9, "a":10, ".":11}

def test_create_inverse_index(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})

    inverse_index = create_inverse_index(index)
    assert create_inverse_index(inverse_index) == index

def test_write_read_index(data2, tmpdir):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})

    write_index(tmpdir + "/", index)
    assert read_index(tmpdir + "/") == index

def test_instance(data2):
    assert Instance().length == 0

    instance = Instance()
    instance[FORM] = None
    assert instance.form == None
    instance.lemma = None
    assert LEMMA in instance
    del instance.form
    assert FORM not in instance

    with pytest.raises(AttributeError):
        instance.unknown
    with pytest.raises(AttributeError):
        del instance.unknown
    instance.unknown = None
    assert instance.unknown == None

def test_map_to_instances(data1, data2):
    sentences = list(read_conllu(data2, skip_empty=True, skip_multiword=True))
    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    inverse_index = create_inverse_index(index)

    instances = list(map_to_instances(sentences, index))

    assert list(map_to_sentences(instances, inverse_index)) == sentences
    assert [sentence.to_instance(index).to_sentence(inverse_index) for sentence in sentences] == sentences

def test_tokens(data2):
    sentences = list(read_conllu(data2, skip_empty=True, skip_multiword=True))
    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    instances = list(map_to_instances(sentences, index))
    tokens = list(instances[0].tokens())

    assert tokens[0] == {f : instances[0][f][0] for f in instances[0].keys()}
    assert len(tokens[0]) == len(instances[0].keys())

    tokens[0][FORM] = -1
    assert instances[0][FORM][0] == -1

    with pytest.raises(TypeError):
        del tokens[0][FORM]

    with pytest.raises(KeyError):
        tokens[0][ID] = 0

def test_normalize(data4):
    sentences = list(read_conllu(data4, skip_empty=False, skip_multiword=False))
    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    instances = list(map_to_instances(sentences, index))

    assert sentences[0][5][FORM_NORM] == NUM_NORM and sentences[0][5][LEMMA_NORM] == NUM_NORM
    assert FORM_NORM_CHARS not in sentences[0][5] and LEMMA_NORM_CHARS not in sentences[0][5]
    assert instances[0][FORM_NORM_CHARS][5] == None and instances[0][LEMMA_NORM_CHARS][5] == None    

    inverse_index = create_inverse_index(index)
    assert list(map_to_sentences(instances, inverse_index)) == sentences

def test_copy(data2):
    token = Token()
    token[ID] = multiword_id(1,2)
    token[FORM] = "vámonos"
    assert token.copy() == token

    sentences = list(read_conllu(data2, skip_empty=True, skip_multiword=True))
    assert sentences == list([sentence.copy() for sentence in sentences])
    
    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    instances = list(map_to_instances(sentences, index))
    assert instances == list([instance.copy() for instance in instances])

def test_is_projective(data2, data5):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    assert [sentence.is_projective() for sentence in sentences] == [True, True]
    assert [sentence.to_tree().is_projective() for sentence in sentences] == [True, True]
    assert [sentence.is_projective(return_arcs=True) for sentence in sentences] == [[], []]

    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    instances = list(map_to_instances(sentences, index))

    assert [instance.is_projective() for instance in instances] == [True, True]
    assert [instance.to_tree().is_projective() for instance in instances] == [True, True]
    assert [instance.is_projective(return_arcs=True) for instance in instances] == [[], []]

    sentences = list(read_conllu(data5, skip_empty=False, skip_multiword=False))
    assert sentences[0].is_projective() == False
    assert sentences[0].to_tree().is_projective() == False
    assert sentences[0].is_projective(return_arcs=True) ==[(3, 6), (6, 7)]

    index = create_index(sentences, fields=set(FIELDS)-{ID, HEAD})
    instances = list(map_to_instances(sentences, index))

    assert instances[0].is_projective() == False
    assert instances[0].to_tree().is_projective() == False
    assert instances[0].is_projective(return_arcs=True) ==[(3, 6), (6, 7)]

if __name__ == "__main__":
    pass
