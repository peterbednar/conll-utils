import os
import h5py

from . import Instance

def write(file, data, format, **kwargs):
    driver = _get_driver(format)
    driver.write(file, data, **kwargs)

def read(file, format, **kwargs):
    driver = _get_driver(format)
    return driver.read(file, **kwargs)

class _TextDriver(object):

    def write(self, file, data, end='\n'):
        if isinstance(file, (str, os.PathLike)):
            file = open(file, 'wt', encoding='utf-8')
        with file:
            for line in data:
                print(file, line, end=end)
    
    def read(self, file):
        if isinstance(file, (str, os.PathLike)):
            file = open(file, 'rt', encoding='utf-8')
        with file:
            for line in file:
                yield line

class _HDF5Driver(object):

    EMPTY_COMMENT_VALUE = 0

    def write(self, file, data, write_comments=True):
        with h5py.File(file, 'w', track_order=True) as f:
            for i, instance in enumerate(data):
                group = f.create_group(str(i), track_order=True)
                if write_comments:
                    self._write_metadata(group, instance)
                self._write_data(group, instance)
    
    def _write_metadata(self, group, instance):
        if isinstance(instance.metadata, dict):
            for atr, val in instance.metadata.items():
                group.attrs[atr] = val if val is not None else self.EMPTY_COMMENT_VALUE

    def _write_data(self, group, instance):
        for field, array in enumerate(instance):
            group.create_dataset(field, array)

    def read(self, file, read_comments=True):
        with h5py.File(file, 'r') as f:
            for key in f.keys():
                group = f[key]
                instance = Instance()
                if read_comments:
                    self._read_metadata(group, instance)
                self._read_data(group, instance)
                yield instance

    def _read_metadata(self, group, instance):
        instance.metadata = {attr : val if val != self.EMPTY_COMMENT_VALUE else None for attr, val in group.attrs.items()}

    def _read_data(self, group, instance):
        for field, array in group.items():
            instance[field] = array

_DRIVERS = {'txt': _TextDriver(), 'hdf5' : _HDF5Driver()}

def _get_driver(format):
    driver = _DRIVERS.get(format)
    if driver is None:
        raise ValueError(f'Unsupported file format {format}.')
    return driver
