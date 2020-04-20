
import re
import numpy as np

from . import Sentence, Token
from . import read_conllu, write_conllu, create_index

def pipe(source=None, *args):
    pipe = Pipeline(source)
    for p in args:
        pipe.pipe(p)
    return pipe

class Pipeline(object):

    def __init__(self, source=None):
        self._pipeline = _Pipe(source)

    @property
    def token(self):
        opr = self._prev_opr()
        if isinstance(opr, TokenPipeline):
            return opr
        opr = TokenPipeline(self)
        self._append_opr(opr)
        return opr

    def filter(self, f):
        self._append_opr(lambda s: s if f(s) else None)
        return self

    def only_projective(self, projective=True):
        self.filter(lambda s: s.is_projective() == projective)
        return self
    
    def map(self, f):
        self._append_opr(f)
        return self

    def text(self):
        self.map(lambda s: s.text)
        return self

    def to_instance(self, index):
        self.map(lambda s: s.to_instance(index))
        return self

    def to_conllu(self):
        self.map(lambda s: s.to_conllu())
        return self

    def from_conllu(self, s):
        self._pipeline.set_source(Sentence.from_conllu(s, multiple=True))
        return self

    def read_conllu(self, filename):
        self._pipeline.set_generator(lambda: read_conllu(filename))
        return self

    def write_conllu(self, filename):
        write_conllu(filename, self)

    def create_index(self, fields=None, min_frequency=1):
        return create_index(self, fields, min_frequency)

    def print(self):
        for s in self():
            print(s)

    def collect(self, l=None):
        if l is None:
            return list(self)
        l.extend(self)
        return l

    def pipe(self, p):
        self._pipeline = _Pipe(self._pipeline, pipe=p)
        return self

    def stream(self, max_size=None):
        self.pipe(lambda source: _stream(source, max_size))
        return self

    def shuffle(self, buffer_size=1024, random=np.random):
        self.pipe(lambda source: _shuffle(source, buffer_size, random))
        return self

    def batch(self, batch_size=100):
        self.pipe(lambda source: _batch(source, batch_size))
        return self

    def __call__(self, source=None):
        return self._pipeline.iterate(source)

    def __iter__(self):
        return self._pipeline.iterate(None)

    def _prev_opr(self):
        return self._pipeline.operations[-1] if self._pipeline.operations else None

    def _append_opr(self, opr):
        self._pipeline.operations.append(opr)

def _stream(source, max_size):
    i = 0
    while True:
        prev = i
        for data in source:
            if max_size is None or i < max_size:
                yield data
                i += 1
            else:
                return
        if prev == i:
            return

def _shuffle(source, buffer_size, random):
    buffer = []
    for data in source:
        if len(buffer) < buffer_size:
            buffer.append(data)
        else:
            i = random.randint(0, len(buffer))
            elm = buffer[i]
            buffer[i] = data
            yield elm
    random.shuffle(buffer)
    for elm in buffer:
        yield elm

def _batch(source, batch_size):
    batch = []
    for data in source:
        if len(batch) < batch_size:
            batch.append(data)
        else:
            yield batch
            batch = [data]
    if batch:
        yield batch

class _Pipe(object):

    def __init__(self, source=None, generator=None, pipe=None):
        self.source = source
        self.generator = generator
        self.pipe = pipe
        self.operations = []

    def set_source(self, source):
        self._check_source()
        self.source = source

    def set_generator(self, generator):
        self._check_source()
        self.generator = generator

    def _check_source(self):
        if self.operations:
            raise RuntimeError('Source must be the first operation.')

        if self.source is not None or self.generator is not None:
            raise RuntimeError('Source is already set.')

    def source_iterator(self, source):
        if source is None:
            source = self.source

        if source is None:
            if self.generator is None:
                raise RuntimeError('No source defined.')
            return self.generator()

        return self.pipe(source) if self.pipe is not None else source

    def iterate(self, source=None):
        for data in self.source_iterator(source):
            for opr in self.operations:
                data = opr(data)
                if data is None:
                    break
            if data is not None:
                yield data

    def __iter__(self):
        return self.iterate(self.source)

class TokenPipeline(object):

    def __init__(self, pipeline):
        self.operations = []
        self._pipeline = pipeline

    def filter(self, f):
        self.operations.append(lambda t: t if f(t) else None)
        return self
    
    def map(self, f):
        self.operations.append(f)
        return self

    def only_words(self):
        self.filter(lambda t: not (t.is_empty or t.is_multiword))
        return self

    def lowercase(self, field, to=None):
        self.map_field(field, lambda s: s.lower(), to)
        return self

    def replace(self, field, regex, value, to=None):
        if isinstance(regex, str):
            regex = re.compile(regex)
        self.map_field(field, lambda s: value if regex.match(s) else s, to)
        return self

    def filter_field(self, field, f):
        self.map_field(field, lambda s: s if f(s) else None)
        return self

    def map_field(self, field, f, to=None):
        if to is None:
            to = field
        def _map_field(t):
            if field in t:
                value = f(t[field])
                if value is not None:
                    t[to] = value
                else:
                    del t[to]
            return t
        self.operations.append(_map_field)
        return self

    def upos_feats(self, to='upos_feats'):
        def _upos_feats(t):
            upos = t.get('upos')
            feats = t.get('feats')
            if upos:
                tag = f"POS={upos}|{feats}" if feats else f"POS={upos}"
            else:
                tag = feats
            if tag:
                t[to] = tag
            return t
        self.map(_upos_feats)
        return self

    def __call__(self, data):
        if isinstance(data, Token):
            for opr in self.operations:
                data = opr(data)
                if data is None:
                    return None
        elif isinstance(data, Sentence):
            i = 0
            for token in data:
                for opr in self.operations:
                    token = opr(token)
                    if token is not None:
                        data[i] = token
                        i += 1
            del data[i:]
        return data
