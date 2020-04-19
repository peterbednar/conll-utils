import os
import pytest
from io import StringIO

from conllutils.pipeline import *

class _StringIO(StringIO):

    def close(self):
        pass

    def release(self):
        super().close()

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
def _data_filename(name):
    return os.path.join(os.path.dirname(__file__), name)

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

def test_filter():
    p = pipe(range(10)).filter(lambda x: x < 5)
    assert p.collect() == [0, 1, 2, 3, 4]

def test_map():
    p = pipe(range(10)).map(lambda x: 2*x)
    assert p.collect() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

def test_pipe():
    p = pipe(range(10), pipe().filter(lambda x: x < 5), pipe(range(10)).map(lambda x: x*2))
    assert p.collect() == [0, 2, 4, 6, 8]

    with pytest.raises(RuntimeError):
        pipe().collect()

def test_print():
    pipe(range(10)).print()

def test_stream():
    p = pipe(range(10)).filter(lambda x: x < 5).stream(10)
    assert p.collect()  == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

def test_batch():
    p = pipe(range(10)).filter(lambda x: x < 5).stream(10).batch(3)
    assert p.collect() == [[0, 1, 2], [3, 4, 0], [1, 2, 3], [4]]

def test_shuffle():
    np.random.seed(1)
    p = pipe(range(10)).filter(lambda x: x < 5).stream(10).shuffle(5)
    assert p.collect()  == [3, 4, 0, 1, 0, 2, 4, 3, 1, 2]

def test_from_conllu():
    p = pipe().from_conllu(_DATA1_CONLLU)
    assert [[t.form for t in s] for s in p.collect()] == [['vámonos', 'vamos', 'nos', 'al', 'a', 'el', 'mar'],
                                                          ['Sue', 'likes', 'coffee', 'and', 'Bill', 'likes', 'tea']]

    with pytest.raises(RuntimeError):
        pipe().map(lambda x: x).from_conllu(_DATA1_CONLLU)

    with pytest.raises(RuntimeError):
        pipe(range(10)).from_conllu(_DATA1_CONLLU)

def test_to_conllu():
    p = pipe().from_conllu(_DATA1_CONLLU).to_conllu()
    s = p.collect()
    assert s[0] + '\n\n' + s[1] + '\n\n' == _DATA1_CONLLU

def test_read_conllu(data1):
    p = pipe().read_conllu(data1)
    assert [[t.form for t in s] for s in p.collect()] == [['vámonos', 'vamos', 'nos', 'al', 'a', 'el', 'mar'],
                                                          ['Sue', 'likes', 'coffee', 'and', 'Bill', 'likes', 'tea']]

def test_read_conllu(data1):
    s = _StringIO()
    pipe().read_conllu(data1).write_conllu(s)
    assert s.getvalue() == _DATA1_CONLLU
    s.release()

_NUM_REGEX = re.compile(r"[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+")
NUM_NORM = u"__number__"

if __name__ == "__main__":
    p = pipe().from_conllu(_DATA1_CONLLU).to_conllu()
    s = _DATA1_CONLLU.split("\n\n")

    print(s)
