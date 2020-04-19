from conllutils.pipeline import *

def test_filter():
    p = pipe(range(10)).filter(lambda x: x < 5)
    assert p.collect() == [0, 1, 2, 3, 4]

def test_map():
    p = pipe(range(10)).map(lambda x: 2*x)
    assert p.collect() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

def test_pipe():
    p = pipe(range(10), pipe().filter(lambda x: x < 5), pipe(range(10)).map(lambda x: x*2))
    assert p.collect() == [0, 2, 4, 6, 8]

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

_NUM_REGEX = re.compile(r"[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+")
NUM_NORM = u"__number__"

if __name__ == "__main__":
    p = pipe(range(10)).filter(lambda x: x < 5).stream(10)

