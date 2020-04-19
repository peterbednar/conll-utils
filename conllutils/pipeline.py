
import re

from . import read_conllu, write_conllu, create_index

class Pipeline(object):

    def __init__(self, source=None):
        self.source = source
        self.operations = []
        self._pipeline = self

    @property
    def token(self):
        opr = self._prev_opr()
        if isinstance(opr, TokenPipeline):
            return opr
        opr = TokenPipeline()
        self._append_opr(opr)
        return opr

    def filter(self, f):
        self._append_opr(lambda s: s if f(s) else None)
        return self
    
    def map(self, f):
        self._append_opr(f)
        return self

    def pipe(self, pipeline):
        self._pipeline = Pipeline(pipeline(self._pipeline))
        return self

    def map_to_instance(self, index):
        self.map(lambda s: s.to_instance(index))
        return self

    def only_projective(self, projective=True):
        self.filter(lambda s: s.is_projective() == projective)
        return self

    def read_conllu(self, filename):
        self._set_source(lambda: read_conllu(filename))
        return self

    def write_conllu(self, filename):
        write_conllu(filename, self)

    def create_index(self):
        return create_index(self)

    def collect(self):
        return list(self)

    def __call__(self, source=None):
        if source is None:
            source = self._iter_source()
        
        for sentence in source:
            for opr in self._pipeline.operations:
                sentence = opr(sentence)
                if sentence is None:
                    break
            if sentence is not None:
                yield sentence

    def __iter__(self):
        return self()

    def _iter_source(self):
        source = self._pipeline.source
        if source == None:
            raise RuntimeError('No source defined.')
        try:
            return iter(source)
        except TypeError:
            return source()

    def _set_source(self, source):
        self._pipeline.source = source

    def _prev_opr(self):
        return self._pipeline.operations[-1] if self._pipeline.operations else None

    def _append_opr(self, opr):
        self._pipeline.operations.append(opr)

class TokenPipeline(object):

    def __init__(self):
        self.operations = []

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
        self.map_field(lambda s: s.lower(), field, to)
        return self

    def replace(self, regex, value, field, to=None):
        if isinstance(regex, str):
            regex = re.compile(regex)
        self.map_field(lambda s: value if regex.match(s) else s, field, to)
        return self

    def filter_field(self, f, field):
        self.map_field(lambda s: s if f(s) else None, field)
        return self

    def map_field(self, f, field, to=None):
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

    def __call__(self, sentence):
        i = 0
        for token in sentence:
            for opr in self.operations:
                token = opr(token)
                if token is not None:
                    sentence[i] = token
                    i += 1
        del sentence[i:]
        return sentence

_NUM_REGEX = re.compile(r"[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+")
NUM_NORM = u"__number__"

if __name__ == "__main__":
    pass
