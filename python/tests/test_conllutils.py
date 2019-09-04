import os
import pytest
import random
from io import StringIO

from conllutils import *
from conllutils import _MULTIWORD, _EMPTY

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
1	vamos	ir	_	_	_	_	_	_	_
2	nos	nosotros	_	_	_	_	_	_	_
3-4	al	_	_	_	_	_	_	_	_
3	a	a	_	_	_	_	_	_	_
4	el	el	_	_	_	_	_	_	_
5	mar	mar	_	_	_	_	_	_	_

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

def test_token():
    token = Token()
    assert str(token) == "<None,None,None>"
    token[ID] = (1,2,_MULTIWORD)
    token[FORM] = "vámonos"
    assert str(token) == "<1-2,vámonos,None>"

def test_dependency_tree(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))
    tree0 = sentences[0].as_tree()    
    assert tree0.root is not None
    assert str(tree0) == "<<2,buy,VERB>,root,[<<1,They,PRON>,nsubj,[]>, <<4,sell,VERB>,conj,[<<3,and,CONJ>,cc,[]>]>, <<5,books,NOUN>,obj,[]>, <<6,.,PUNCT>,punct,[]>]>"

    nodes = []
    tree0.visit(lambda node: nodes.append(node))
    assert [node.is_root for node in nodes] == [True, False, False, False, False, False]

def test_read_conllu(data1):
    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))
    assert sentences == [[
            _fields((1,2,_MULTIWORD), "vámonos"),
            _fields(1, "vamos", "ir"),
            _fields(2, "nos", "nosotros"),
            _fields((3,4,_MULTIWORD), "al"),
            _fields(3, "a", "a"),
            _fields(4, "el", "el"),
            _fields(5, "mar", "mar")
        ], [
            _fields(1, "Sue", "Sue"),
            _fields(2, "likes", "like"),
            _fields(3, "coffee", "coffee"),
            _fields(4, "and", "and"),
            _fields(5, "Bill", "Bill"),
            _fields((5,1,_EMPTY), "likes", "like"),
            _fields(6, "tea", "tea"),
    ]]

def test_parse_deps_feats(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False, parse_deps=True, parse_feats=True, upos_feats=False, normalize=None, split=None))
    assert sentences[0][0][FEATS] == {"Case":"Nom", "Number":"Plur"}
    assert sentences[0][0][DEPS] == [(2, "nsubj"), (4, "nsubj")]
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

    sentences = list(read_conllu(data2, parse_deps=True, parse_feats=True, upos_feats=True, normalize=None, split=None))
    assert [sentences[0][i][UPOS_FEATS] for i in range(len(sentences[0]))] == [
            "POS=PRON|Case=Nom|Number=Plur",
            "POS=VERB|Number=Plur|Person=3|Tense=Pres",
            "POS=CONJ",
            "POS=VERB|Number=Plur|Person=3|Tense=Pres",
            "POS=NOUN|Number=Plur",
            "POS=PUNCT"
        ]

def test_empty_multiword(data1):
    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False))

    assert [token.is_multiword for token in sentences[0]] == [True, False, False, True, False, False, False]
    assert [token.is_empty for token in sentences[0]] == [False, False, False, False, False, False, False]

    assert [token.is_multiword for token in sentences[1]] == [False, False, False, False, False, False, False]
    assert [token.is_empty for token in sentences[1]] == [False, False, False, False, False, True, False]

    assert [sentences[0].get(i)[FORM] for i in range(1, 6)] == ["vamos", "nos", "a", "el", "mar"]
    assert [sentences[1].get(i)[FORM] for i in range(1, 7)] == ["Sue", "likes", "coffee", "and", "Bill", "tea"]

    assert sentences[0].get((1,2,_MULTIWORD))[FORM] == "vámonos"
    assert sentences[1].get((5,1,_EMPTY))[FORM] == "likes"

    with pytest.raises(IndexError):
        sentences[0].get(0)

    with pytest.raises(IndexError):
        sentences[0].get(6)

    with pytest.raises(IndexError):
        sentences[0].get((1,3,_MULTIWORD))

    with pytest.raises(IndexError):
        sentences[0].get((1,2,_EMPTY))

    sentences = list(read_conllu(data1, skip_empty=True, skip_multiword=True))
    for sentence in sentences:
        for token in sentence:
            assert (not token.is_empty) and (not token.is_multiword)

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

def test_decode_conllu():
    sentences = list(decode_conllu(_DATA1_CONLLU, skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))
    assert sentences == [[
            _fields((1,2,_MULTIWORD), "vámonos"),
            _fields(1, "vamos", "ir"),
            _fields(2, "nos", "nosotros"),
            _fields((3,4,_MULTIWORD), "al"),
            _fields(3, "a", "a"),
            _fields(4, "el", "el"),
            _fields(5, "mar", "mar")
        ], [
            _fields(1, "Sue", "Sue"),
            _fields(2, "likes", "like"),
            _fields(3, "coffee", "coffee"),
            _fields(4, "and", "and"),
            _fields(5, "Bill", "Bill"),
            _fields((5,1,_EMPTY), "likes", "like"),
            _fields(6, "tea", "tea"),
    ]]

def test_encode_conllu():
    sentences = list(decode_conllu(_DATA1_CONLLU, skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))
    assert encode_conllu(sentences) == _DATA1_CONLLU

def test_create_dictionary(data2):
        sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
        dictionary = create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD})

        assert dictionary.keys() == set(FIELDS)-{ID, HEAD}
        assert dictionary[FORM] == {"They":1, "buy":1, "and":1, "sell":1, "books":1, ".":2, "I":1, "have":1, "no":1, "clue":1}
        assert dictionary[FORM_NORM_CHARS] == {"e":4, "l":3, "o":3, "h":2, "y":2, "b":2, "u":2, "a":2, "n":2, "s":2, ".":2, "t":1, "d":1, "k":1, "i":1, "v":1, "c":1}

def test_create_index(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    dictionary = create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD})
    
    index = create_index(dictionary)
    assert [list(index[f].values()) for f in index.keys()] == [list(range(1, len(index[f])+1)) for f in index.keys()]
    assert [index[f].keys() for f in index.keys()] == [dictionary[f].keys() for f in index.keys()]

    index = create_index(dictionary, min_frequency=2)
    assert index[FORM] == {".":1}
    assert index[FORM_NORM_CHARS] == {"e":1, "l":2, "o":3, "h":4, "y":5, "b":6, "u":7, "a":8, "n":9, "s":10, ".":11}

def test_create_inverse_index(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    index = create_index(create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD}))

    inverse_index = create_inverse_index(index)
    assert create_inverse_index(inverse_index) == index

def test_write_read_index(data2, tmpdir):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    index = create_index(create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD}))

    write_index(tmpdir + "/", index)
    assert read_index(tmpdir + "/") == index

def test_count_frequency(data2):
    sentences = list(read_conllu(data2, skip_empty=False, skip_multiword=False))
    dictionary = create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD})
    index = create_index(dictionary)

    frequencies = count_frequency(sentences, index)
    for f in index.keys():
        assert {index[f][k]:v for k,v in dictionary[f].items()} == frequencies[f]

def test_map_to_instances(data2):
    sentences = list(read_conllu(data2, skip_empty=True, skip_multiword=True))
    index = create_index(create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD}))
    inverse_index = create_inverse_index(index)

    instances = list(map_to_instances(sentences, index))
    assert list(map_to_sentences(instances, inverse_index)) == sentences

def test_iterate_tokens(data2):
    sentences = list(read_conllu(data2, skip_empty=True, skip_multiword=True))
    index = create_index(create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD}))
    instances = list(map_to_instances(sentences, index))
    tokens = list(iterate_tokens(instances))

    assert len(tokens) == sum(len(sentence) for sentence in sentences)
    assert tokens[0] == {f : instances[0][f][0] for f in instances[0].keys()}

def test_normalize(data4):
    sentences = list(read_conllu(data4, skip_empty=False, skip_multiword=False))
    index = create_index(create_dictionary(sentences, fields=set(FIELDS)-{ID, HEAD}))
    instances = list(map_to_instances(sentences, index))

    assert sentences[0][5][FORM_NORM] == NUM_NORM and sentences[0][5][LEMMA_NORM] == NUM_NORM
    assert FORM_NORM_CHARS not in sentences[0][5] and LEMMA_NORM_CHARS not in sentences[0][5]
    assert instances[0][FORM_NORM_CHARS][5] == None and instances[0][LEMMA_NORM_CHARS][5] == None    

    inverse_index = create_inverse_index(index)
    assert list(map_to_sentences(instances, inverse_index)) == sentences

def test_shuffled_stream():
    random.seed(1)
    data = list(range(5))
    assert list(shuffled_stream(data, size=10)) == [2, 3, 4, 0, 1, 2, 4, 3, 1, 0]

    data = list(range(5))
    random.seed(1)

    i = 0
    values = []
    for value in shuffled_stream(data):
        values.append(value)
        i += 1
        if i >= 10:
            break
    assert values == [2, 3, 4, 0, 1, 2, 4, 3, 1, 0]
