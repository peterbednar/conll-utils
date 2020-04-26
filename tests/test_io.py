import os
import pytest
import numpy as np

from conllutils.pipeline import pipe

def _data_filename(name):
    return os.path.join(os.path.dirname(__file__), name)

def _read_file(name):
    with open(name, "rt", encoding="utf-8") as f:
        return f.read()

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

def test_read_write_file():
    with pytest.raises(ValueError):
        pipe().read_file('temp', 'unknown').collect()

def test_collu(data2, tmp_path):
    sentences = pipe().read_conllu(data2).collect()
    filename = tmp_path / 'data.conllu'

    pipe(sentences).write_file(filename, 'conllu')
    assert pipe().read_file(filename, 'conllu').collect() == sentences

def test_txt(tmp_path):
    lines1 = ["text"] * 10
    filename = tmp_path / 'data.txt'
    pipe(lines1).write_file(filename, 'txt')
    lines2 = pipe().read_file(filename, 'txt').collect()

    assert lines2 == [l + '\n' for l in lines1]

def test_hdf5(data2, data3, tmp_path):
    index = pipe().read_conllu(data2).create_index()
    instances1 = pipe().read_conllu(data2).to_instance(index).collect()

    filename = tmp_path / 'data.hdf5'
    pipe(instances1).write_file(filename, 'hdf5')
    instances2 = pipe().read_file(filename, 'hdf5').collect()

    for ins1, ins2 in zip(instances1, instances2):
        equal_instance(ins1, ins2)

    index = pipe().read_conllu(data3).create_index()
    instances1 = pipe().read_conllu(data3).to_instance(index).collect()

    filename = tmp_path / 'data.hdf5'
    pipe(instances1).write_file(filename, 'hdf5')
    instances2 = pipe().read_file(filename, 'hdf5').collect()

    for ins1, ins2 in zip(instances1, instances2):
        equal_instance(ins1, ins2)

def equal_instance(ins1, ins2):
    assert ins1.metadata == ins2.metadata
    assert ins1.keys() == ins2.keys()

    for field in ins1.keys():
        assert np.array_equal(ins1[field], ins2[field])

if __name__ == "__main__":
    pass
