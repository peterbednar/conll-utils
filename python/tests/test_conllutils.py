import os
import pytest
from io import StringIO

from conllutils import *

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

def test_token():
    token = Token()
    assert str(token) == "<None,None,None>"
    token[ID] = (1,2,MULTIWORD)
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
            _fields((1,2,MULTIWORD), "vámonos"),
            _fields(1, "vamos", "ir"),
            _fields(2, "nos", "nosotros"),
            _fields((3,4,MULTIWORD), "al"),
            _fields(3, "a", "a"),
            _fields(4, "el", "el"),
            _fields(5, "mar", "mar")
        ], [
            _fields(1, "Sue", "Sue"),
            _fields(2, "likes", "like"),
            _fields(3, "coffee", "coffee"),
            _fields(4, "and", "and"),
            _fields(5, "Bill", "Bill"),
            _fields((5,1,EMPTY), "likes", "like"),
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

def test_empty_multiword(data1):
    sentences = list(read_conllu(data1, skip_empty=False, skip_multiword=False))

    assert [token.is_multiword for token in sentences[0]] == [True, False, False, True, False, False, False]
    assert [token.is_empty for token in sentences[0]] == [False, False, False, False, False, False, False]

    assert [token.is_multiword for token in sentences[1]] == [False, False, False, False, False, False, False]
    assert [token.is_empty for token in sentences[1]] == [False, False, False, False, False, True, False]

    assert [sentences[0].get(i)[FORM] for i in range(1, 6)] == ["vamos", "nos", "a", "el", "mar"]
    assert [sentences[1].get(i)[FORM] for i in range(1, 7)] == ["Sue", "likes", "coffee", "and", "Bill", "tea"]

    assert sentences[0].get((1,2,MULTIWORD))[FORM] == "vámonos"
    assert sentences[1].get((5,1,EMPTY))[FORM] == "likes"

    with pytest.raises(IndexError):
        sentences[0].get(0)

    with pytest.raises(IndexError):
        sentences[0].get(6)

    with pytest.raises(IndexError):
        sentences[0].get((1,3,MULTIWORD))

    with pytest.raises(IndexError):
        sentences[0].get((1,2,EMPTY))

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

    input = _read_file(data3)
    sentences = list(read_conllu(data3, skip_empty=False, skip_multiword=False, parse_deps=True, parse_feats=True, upos_feats=False, normalize=None, split=None))
    output = _StringIO()
    write_conllu(output, sentences)
    assert output.getvalue() == input
    output.release()
