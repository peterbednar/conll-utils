import os

from . import Instance, write_conllu, read_conllu

def write_file(file, data, format, **kwargs):
    driver = _get_driver(format)
    driver.write(file, data, **kwargs)

def read_file(file, format, **kwargs):
    driver = _get_driver(format)
    return driver.read(file, **kwargs)

class _CoNLLUDriver(object):

    def write(self, file, data, **kwargs):
        write_conllu(file, data, **kwargs)
    
    def read(self, file, **kwargs):
        return read_conllu(file, **kwargs)

class _TextDriver(object):

    def write(self, file, data, end='\n'):
        if isinstance(file, (str, os.PathLike)):
            file = open(file, 'wt', encoding='utf-8')
        with file:
            for line in data:
                print(line, file=file, end=end)
    
    def read(self, file):
        if isinstance(file, (str, os.PathLike)):
            file = open(file, 'rt', encoding='utf-8')
        with file:
            for line in file:
                yield line

_DRIVERS = {'txt': _TextDriver(), 'conllu': _CoNLLUDriver()}

def _get_driver(format):
    driver = _DRIVERS.get(format)
    if driver is None:
        raise ValueError(f'Unsupported file format {format}.')
    return driver

try:
    import h5py
except ImportError:
    # h5py is not installed (HDF5 format is not supported)
    pass
else:

    from . import _metadata_to_str, _parse_metadata

    class _HDF5Driver(object):

        METADATA_ATTR = 'comments'

        def write(self, file, data, write_comments=True):
            with h5py.File(file, 'w', track_order=True) as f:
                for i, instance in enumerate(data):
                    group = f.create_group(str(i))
                    if write_comments:
                        self._write_metadata(group, instance)
                    self._write_data(group, instance)
    
        def _write_metadata(self, group, instance):
            if isinstance(instance.metadata, dict):
                group.addrs[self.METADATA_ATTR] = '\n'.join(_metadata_to_str(instance.metadata))

        def _write_data(self, group, instance):
            for field, array in instance.items():
                group.create_dataset(field, data=array)

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
            if self.METADATA_ATTR in group.attrs:
                instance.metadata = _parse_metadata(group.attrs[self.METADATA_ATTR].splitlines())

        def _read_data(self, group, instance):
            for field, array in group.items():
                instance[field] = array.value

    _DRIVERS['hdf5'] = _HDF5Driver()
