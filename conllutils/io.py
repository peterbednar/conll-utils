import h5py

from . import Instance

def write(stream, file, format, **kwargs):
    driver = _get_driver(format)
    driver.write(stream, file, **kwargs)

def read(file, format, **kwargs):
    driver = _get_driver(format)
    return driver.read(file, **kwargs)

class _HDF5Driver(object):

    def write(self, stream, file):
        with h5py.File(file, 'w', track_order=True) as f:
            for i, data in enumerate(stream):
                group = f.create_group(str(i), track_order=True)

                self._write_metadata(group, data)
                self._write_data(group, data)
    
    @staticmethod
    def _write_metadata(group, instance):
        if isinstance(instance.metadata, dict):
            for atr, val in instance.metadata.items():
                group.attrs[atr] = val

    @staticmethod
    def _write_data(group, instance):
        for field, array in enumerate(instance):
            group.create_dataset(field, array)

    def read(self, file):
        with h5py.File(file, 'r') as f:
            for key in f.keys():
                group = f[key]
                instance = Instance()

                self._read_metadata(group, instance)
                self._read_data(group, instance)
                yield instance

    @staticmethod
    def _read_metadata(group, instance):
        metadata = {group.attrs.items()}
        instance.metadata = metadata

    @staticmethod
    def _read_data(group, instance):
        instance.update({group.items()})

_DRIVERS = {'hdf5' : _HDF5Driver()}

def _get_driver(format):
    driver = _DRIVERS.get(format)
    if driver is None:
        raise ValueError(f'Unsupported file format {format}.')
    return driver
