import os
import pytest

from conllutils import *

def _data_filename(name):
    return os.path.join(os.path.dirname(__file__), name)

def _data_file(name):
    return open(_data_filename(name), "rt", encoding="utf-8")

def _fields(*values):
    return {i : v for i, v in enumerate(values) if v is not None}

def test_read_conllu():
    sentences = list(read_conllu(_data_filename("data1.conllu"), skip_empty=False, skip_multiword=False, upos_feats=False, normalize=None, split=None))

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

def test_empty_multiword():
    sentences = list(read_conllu(_data_filename("data1.conllu"), skip_empty=False, skip_multiword=False))

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

    sentences = list(read_conllu(_data_filename("data1.conllu"), skip_empty=True, skip_multiword=True))
    for sentence in sentences:
        for token in sentence:
            assert (not token.is_empty) and (not token.is_multiword)

