import os
import pytest
from io import StringIO

import numpy as np

from conllutils import FORM, FIELDS, ID, HEAD
from conllutils import pipe, create_inverse_index

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

def test_count():
    assert pipe(range(5)).count() == 5
    assert pipe([]).count() == 0

def test_first():
    assert pipe(range(5)).first() == 0
    assert pipe([]).first() == None
    assert pipe([]).first(0) == 0

def test_collect():
    assert pipe(range(5)).collect(l=[-1]) == list(range(-1, 5))
    assert pipe(range(5)).collect(3, [-1]) == list(range(-1, 3))

def test_print(capsys):
    pipe(range(5)).print()
    pipe(range(5)).print(1)
    captured = capsys.readouterr()
    assert captured.out == '0\n1\n2\n3\n4\n0\n'

def test_stream():
    p = pipe(range(10)).filter(lambda x: x < 5).stream(10)
    assert p.collect()  == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    p = pipe(()).stream(10)
    assert p.collect() == []

def test_batch():
    p = pipe(range(10)).filter(lambda x: x < 5).stream(10).batch(3)
    assert p.collect() == [[0, 1, 2], [3, 4, 0], [1, 2, 3], [4]]

    p = pipe(range(5)).batch(3, size=lambda _: 2)
    assert p.collect() == [[0, 1], [2, 3], [4]]

    p = pipe(range(5)).batch(3, size=lambda _: 3)
    assert p.collect() == [[0], [1], [2], [3], [4]]

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

def test_write_conllu(data1):
    s = _StringIO()
    pipe().read_conllu(data1).write_conllu(s)
    assert s.getvalue() == _DATA1_CONLLU
    s.release()

def test_only_projective(data2, data5):
    p = pipe().read_conllu(data2).only_projective()
    assert [s.is_projective() for s in p.collect()] == [True, True]

    p = pipe().read_conllu(data5).only_projective()
    assert p.collect() == []

    p = pipe().read_conllu(data5).only_projective(False)
    assert [s.is_projective() for s in p.collect()] == [False]

def test_text(data2):
    p = pipe().read_conllu(data2).text()
    assert p.collect() == ['They buy and sell books. ', 'I have no clue. ']

def test_create_index(data2):
    index = pipe().read_conllu(data2).create_index(fields=set(FIELDS)-{ID, HEAD}, min_frequency=2)
    assert index[FORM] == {".":1}

def test_to_instance(data2):
    sentences = pipe().read_conllu(data2).collect()
    index = pipe(sentences).create_index(fields=set(FIELDS)-{ID, HEAD})
    inverse_index = create_inverse_index(index)

    p = pipe().read_conllu(data2).to_instance(index)
    k = pipe(p).to_sentence(inverse_index)

    assert k.collect() == sentences

def test_to_flatten():
    p = pipe(range(10)).batch(3).flatten()
    assert p.collect() == list(range(10))

    p = pipe(range(10)).flatten()
    assert p.collect() == list(range(10))

def test_only_words(data1):
    p = pipe().read_conllu(data1).only_words().text()
    assert p.collect() == ['vamos nos a el mar ', 'Sue likes coffee and Bill tea ']

def test_only_fields(data2):
    sentences = pipe().read_conllu(data2).only_fields({'id', 'form'}).collect()
    assert [[t.keys() for t in s] for s in sentences] == [[{'id', 'form'}] * len(s) for s in sentences]

    sentences = pipe().read_conllu(data2).only_fields('id', 'form').collect()
    assert [[t.keys() for t in s] for s in sentences] == [[{'id', 'form'}] * len(s) for s in sentences]

def test_upos_feats(data2):
    sentences = pipe().read_conllu(data2).upos_feats('new').collect()
    assert [t.get('new') for t in sentences[0]] == [
        'POS=PRON|Case=Nom|Number=Plur',
        'POS=VERB|Number=Plur|Person=3|Tense=Pres',
        'POS=CONJ',
        'POS=VERB|Number=Plur|Person=3|Tense=Pres',
        'POS=NOUN|Number=Plur',
        'POS=PUNCT']

def test_lowercase(data1):
    p = pipe().read_conllu(data1).only_words().lowercase('form').text()
    assert p.collect() == ['vamos nos a el mar '.lower(), 'Sue likes coffee and Bill tea '.lower()]

def test_uppercase(data1):
    p = pipe().read_conllu(data1).only_words().uppercase('form').text()
    assert p.collect() == ['vamos nos a el mar '.upper(), 'Sue likes coffee and Bill tea '.upper()]

def test_replace(data4):
    sentences = pipe().read_conllu(data4).replace('form', r"[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+", '__number__', 'new').collect()
    assert [t.get('new') for t in sentences[0]] == ['Posledná', 'revízia', 'vyšla', 'v', 'roku', '__number__', '.']

    del sentences[0][0].form
    pipe(sentences).replace('form', None, '__missing__', 'new').collect()
    assert [t.get('new') for t in sentences[0]] == ['__missing__', 'revízia', 'vyšla', 'v', 'roku', '__number__', '.']

def test_replace_missing(data4):
    sentences = pipe().read_conllu(data4).collect()
    del sentences[0][0].form
    pipe(sentences).replace_missing('form', '__missing__', 'new').collect()
    assert [t.get('new') for t in sentences[0]] == ['__missing__', None, None, None, None, None, None]

    pipe(sentences).replace_missing('form', None, 'new').collect()
    assert [t.get('new') for t in sentences[0]] == [None, None, None, None, None, None, None]

def test_flatten_tokens(data1):
    tokens = pipe().read_conllu(data1).flatten().only_words().lowercase('form').collect()
    assert [t.form for t in tokens] == ['vamos', 'nos', 'a', 'el', 'mar', 'sue', 'likes', 'coffee', 'and', 'bill', 'tea']

def test_map_field(data1):
    tokens = pipe().read_conllu(data1).flatten().map_field('form', lambda s: None).collect()
    assert [t for t in tokens if 'form' in t] == []

def test_filter_field(data1):
    tokens = pipe().read_conllu(data1).flatten().filter_field('form', lambda s: False).collect()
    assert [t for t in tokens if 'form' in t] == []

def test_map_token(data1):
    sentences = pipe().read_conllu(data1).map_token(lambda t: None).collect()
    assert sentences == [[], []]

def test_filter_token(data1):
    sentences = pipe().read_conllu(data1).filter_token(lambda t: False).collect()
    assert sentences == [[], []]

def test_only_universal_deprel(data4):
    sentences = pipe().read_conllu(data4).collect()
    sentences[0][0].deprel = 'test:test'
    sentences[0][0].deps = '1:test1:test:test|2:test2'
    sentences = pipe(sentences).only_universal_deprel().collect()

    assert sentences[0][0].deprel == 'test'
    assert sentences[0][0].deps == '1:test1|2:test2'
    assert sentences[0][-1].deps == '3:punct'

    sentences = pipe().read_conllu(data4, parse_deps=True).collect()
    sentences[0][0].deps = {(1, 'test1:test'), (2, 'test2')}
    sentences = pipe(sentences).only_universal_deprel().collect()

    assert sentences[0][0].deprel == 'amod'
    assert sentences[0][0].deps == {(1, 'test1'), (2, 'test2')}
    assert sentences[0][-1].deps == {(3, 'punct')}

def test_unwind_feats(data2):
    sentences = pipe().read_conllu(data2).unwind_feats().collect()
    assert [[t.get('feats:Number') for t in s] for s in sentences] == [
        ['Plur', 'Plur', None, 'Plur', 'Plur', None],
        ['Sing', 'Sing', None, 'Sing', None]]
    assert [[t.get('feats:Case') for t in s] for s in sentences] == [
        ['Nom', None, None, None, None, None],
        ['Nom', None, None, None, None]]

def test_remove_fields(data2):
    sentences = pipe().read_conllu(data2).remove_fields({'id', 'form'}).collect()
    assert [['id' in t.keys() for t in s] for s in sentences] == [[False] * len(s) for s in sentences]
    assert [['form' in t.keys() for t in s] for s in sentences] == [[False] * len(s) for s in sentences]

    sentences = pipe().read_conllu(data2).remove_fields('id', 'form').collect()
    assert [['id' in t.keys() for t in s] for s in sentences] == [[False] * len(s) for s in sentences]
    assert [['form' in t.keys() for t in s] for s in sentences] == [[False] * len(s) for s in sentences]

def test_split_chars(data2):
    sentences = pipe().read_conllu(data2).split_chars('form').collect()
    assert [[t['form:chars'] for t in s] for s in sentences] == [[tuple(t.form) for t in s] for s in sentences]

if __name__ == "__main__":
    pass