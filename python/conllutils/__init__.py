import re
import random
import os
from collections import OrderedDict, Counter 
import numpy as np

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, \
FORM_NORM, LEMMA_NORM, FORM_CHARS, LEMMA_CHARS, FORM_NORM_CHARS, LEMMA_NORM_CHARS, UPOS_FEATS = range(17)

EMPTY = 0
MULTIWORD = 1

FIELD_TO_STR = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc",
                "form_norm", "lemma_norm", "form_chars", "lemma_chars", "form_norm_chars", "lemma_norm_chars",
                "upos_feats"]
STR_TO_FIELD = {k : v for v, k in enumerate(FIELD_TO_STR)}

_BASE_FIELDS = [ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]

_CHARS_FIELDS_MAP = {FORM: FORM_CHARS, LEMMA: LEMMA_CHARS, FORM_NORM: FORM_NORM_CHARS, LEMMA_NORM: LEMMA_NORM_CHARS}
_CHARS_FIELDS = set(_CHARS_FIELDS_MAP.values())

def str_to_field(s):
    return STR_TO_FIELD[s.lower()]

def field_to_str(f):
    return FIELD_TO_STR[f]

def id_to_str(id):
    if isinstance(tuple):
        return f"{id[0]}.{id[1]}" if id[2] == EMPTY else f"{id[0]}-{id[1]}"
    else:
        return str(id)

class Token(dict):

    def __init__(self, fields={}):
        super().__init__(fields)

    @property
    def is_empty(self):
        id = self.get(ID)
        return id[2] == EMPTY if isinstance(id, tuple) else False

    @property
    def is_multiword(self):
        id = self.get(ID)
        return id[2] == MULTIWORD if isinstance(id, tuple) else False

class Sentence(list):

    def __init__(self, tokens=[], metadata=None):
        super().__init__(tokens)
        self.tokens = self
        self.metadata = metadata

    def get(self, id):
        start = id[0] if isinstance(id, tuple) else id
        for token in self.tokens[start-1:]:
            if token[ID] == id:
                return token
        raise IndexError(f"token with id {id} not found")

    def as_tree(self):
        return DependencyTree(self)

class Node(object):

    def __init__(self, token, parent):
        self.token = token
        self.parent = parent
        self.children = []

    @property
    def is_root(self):
        return self.parent == None

    @property
    def deprel(self):
        return self.token[DEPREL]

class DependencyTree(object):

    def __init__(self, sentence):
        self.root = DependencyTree.build(sentence)
        self.metadata = sentence.metadata

    @staticmethod
    def build(sentence):
        return DependencyTree._build(sentence, None)

    @staticmethod
    def _build(sentence, parent):
        id = parent.token[ID] if parent is not None else 0
        
        for token in sentence:
            if token[ID] == id:
                node = Node(token, parent)
                DependencyTree._build(sentence, node)
                if parent == None:
                    return node
                else:
                    parent.children.append(node)

        return parent

class Instance(dict):
    
    def __init__(self, length=0, fields={}, metadata=None):
        super().__init__(fields)
        self._length = length
        self.metadata = metadata

    def __len__(self):
        return self._length

_NUM_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
NUM_NORM = u"__number__"

def normalize_lower(field, value):
    if field == FORM or field == LEMMA:
        return value.lower()
    return value

def normalize_default(field, value):
    if field == FORM or field == LEMMA:
        if _NUM_REGEX.match(value):
            return NUM_NORM
        return value.lower()
    return value

def split_default(field, value):
    if (field == FORM_NORM or field == LEMMA_NORM) and value == NUM_NORM:
        return None
    return tuple(value)

def read_conllu(file, skip_empty=True, skip_multiword=True, parse_feats=False, parse_deps=False, upos_feats=True,
                normalize=normalize_default, split=split_default):

    def _parse_sentence(lines, comments):
        tokens = []
        metadata = _parse_metadata(comments)

        for line in lines:
            token = _parse_token(line)
            if skip_empty and token.is_empty:
                continue
            if skip_multiword and token.is_multiword:
                continue
            tokens.append(token)

        return Sentence(tokens, metadata)

    def _parse_metadata(comments):
        return [comment[1:].strip() for comment in comments]

    def _parse_token(line):
        fields = line.split("\t")
        fields = {f : fields[f] for f in range(min(len(fields), MISC + 1))}

        if "." in fields[ID]:
            token_id, index = fields[ID].split(".")
            id = (int(token_id), int(index), EMPTY)
        elif "-" in fields[ID]:
            start, end = fields[ID].split("-")
            id = (int(start), int(end), MULTIWORD)
        else:
            id = int(fields[ID])
        fields[ID] = id

        for f in [FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]:
            if f in fields and fields[f] == "_":
                del(fields[f])

        if upos_feats:
            upos = fields.get(UPOS)
            feats = fields.get(FEATS)
            if upos:
                tag = f"POS={upos}|{feats}" if feats else f"POS={upos}"
            else:
                tag = feats
            if tag:
                fields[UPOS_FEATS] = tag

        if parse_feats and FEATS in fields:
            fields[FEATS] = _parse_feats(fields[FEATS])

        if HEAD in fields:
            fields[HEAD] = int(fields[HEAD])

        if parse_deps and DEPS in fields:
            fields[DEPS] = _parse_deps(fields[DEPS])

        if normalize:
            for (f, n) in [(FORM, FORM_NORM), (LEMMA, LEMMA_NORM)]:
                if f in fields:
                    norm = normalize(f, fields[f])
                    if norm is not None:
                        fields[n] = norm

        if split:
            for (f, ch) in _CHARS_FIELDS_MAP.items():
                if f in fields:
                    chars = split(f, fields[f])
                    if chars is not None:
                        fields[ch] = chars

        return Token(fields)

    def _parse_feats(str):
        feats = OrderedDict()
        for key, value in [feat.split("=") for feat in str.split("|")]:
            if "," in value:
                value = value.split(",")
            feats[key] = value
        return feats

    def _parse_deps(str):
        return list(map(lambda rel: (int(rel[0]), rel[1]), [rel.split(":") for rel in str.split("|")]))

    lines = []
    comments = []
    if isinstance(file, str):
        file = open(file, "rt", encoding="utf-8")
    with file:
        for line in file:
            line = line.rstrip("\r\n")
            if line.startswith("#"):
                comments.append(line)
            elif line.lstrip():
                lines.append(line)
            else :
                if len(lines) > 0:
                    yield _parse_sentence(lines, comments)
                    lines = []
                    comments = []
        if len(lines) > 0:
            yield _parse_sentence(lines, comments)

def write_conllu(file, sentences):

    def _write_metadata(fp, metadata):
        if metadata:
            for comment in metadata:
                print("# " + comment, file=fp)

    def _write_tokens(fp, tokens):
        for token in tokens:
            fields = "\t".join([_field_to_str(token, field) for field in _BASE_FIELDS])
            print(fields, file=fp)

    def _field_to_str(token, field):

        if field == ID:
            return id_to_str(token[ID])

        if field not in token or token[field] is None:
            return "_"

        if field == FEATS:
            return _feats_to_str(token[FEATS])

        if field == DEPS:
            return _deps_to_str(token[DEPS])

        return str(token[field])

    def _feats_to_str(feats):
        if isinstance(feats, str):
            return feats
        feats = [key + "=" + (",".join(value) if isinstance(value, list) else value) for key, value in feats.items()]
        return "|".join(feats)        

    def _deps_to_str(deps):
        if isinstance(deps, str):
            return deps
        deps = [f"{rel[0]}:{rel[1]}" for rel in deps]
        return "|".join(deps)

    if isinstance(file, str):
        file = open(file, "wt", encoding="utf-8")
    with file as fp:
        for sentence in sentences:
            _write_metadata(fp, sentence.metadata)
            _write_tokens(fp, sentence.tokens)
            print(file=fp)

def create_dictionary(sentences, fields={FORM, LEMMA, UPOS, XPOS, FEATS, DEPREL}):
    dic = {f: Counter() for f in fields}
    for sentence in sentences:
        for token in sentence:
            for f in fields:
                s = token.get(f)
                if isinstance(s, (list, tuple)):
                    for ch in s:
                        dic[f][ch] += 1
                else:
                    dic[f][s] += 1
    return dic

def create_index(dic, min_frequency=1):
    index = {f: Counter() for f in dic.keys()}
    for f, c in dic.items():
        ordered = c.most_common()
        min_fq = min_frequency[f] if isinstance(min_frequency, (list, tuple, dict)) else min_frequency
        for i, (s, fq) in enumerate(ordered):
            if fq >= min_fq:
                index[f][s] = i + 1
    return index

def create_inverse_index(index):
    return {f: {v: k for k, v in c.items()} for f, c in index.items()}

INDEX_FILENAME = "{0}index_{1}.txt"

_NONE_TOKEN = u"__none__"

def write_index(index, fields=None, basename=""):
    if fields is None:
        fields = index.keys()
    index = create_inverse_index(index)
    for f in fields:
        filename = INDEX_FILENAME.format(basename, field_to_str(f))
        with open(filename, "wt", encoding="utf-8") as fp:
            c = index[f]
            for i in range(1, len(c) + 1):
                token = c[i]
                if token is None:
                    token = _NONE_TOKEN
                print(token, file=fp)

def read_index(fields=None, basename=""):
    if fields is None:
        fields = range(len(FIELD_TO_STR))
    index = {}
    for f in fields:
        filename = INDEX_FILENAME.format(basename, field_to_str(f))
        if os.path.isfile(filename):
            with open(filename, "rt", encoding="utf-8") as fp:
                index[f] = Counter()
                i = 1
                for line in fp:
                    token = line.rstrip("\r\n")
                    if token == _NONE_TOKEN:
                        token = None
                    index[f][token] = i
                    i += 1
    return index

def count_frequency(sentences, index, fields=None):
    if fields is None:
        fields = index.keys()
    count = {f: Counter() for f in fields}
    for sentence in sentences:
        for token in sentence:
            for f in fields:
                s = token.get(f)
                if isinstance(s, (list, tuple)):
                    for ch in s:
                        i = index[f][ch]
                        count[f][i] += 1
                else:
                    i = index[f][s]
                    count[f][i] += 1
    return count

def map_to_instances(sentences, index, fields=None):
    for sentence in sentences:
        yield map_to_instance(sentence, index, fields)

def map_to_instance(sentence, index, fields=None):
    if fields is None:
        fields = [DEPREL, HEAD] + index.keys()

    l = len(sentence)
    instance = Instance(l)
    instance.metadata = sentence.metadata

    for field in fields:
        dtype = np.object if field in _CHARS_FIELDS else np.int
        array = np.array(l, dtype=dtype)

        for i, token in enumerate(sentence):
            value = token.get(field)
            if field == HEAD:
                array[i] = value
            elif field in _CHARS_FIELDS:
                if value is None:
                    array[i] = None
                else:
                    chars = [index[field][ch] for ch in value]
                    array[i] = np.array(chars, dtype=np.int)
            else:
                array[i] = index[field][value]

        instance[field] = array
    
    return instance

def join_default(field, value):
    return "".join(value)

def map_to_sentences(instances, index, fields=None, join=join_default):
    for instance in instances:
        yield map_to_sentence(instance, index, fields, join)

def map_to_sentence(instance, index, fields=None, join=join_default):
    if fields is None:
        fields = instance.keys()

    tokens = []
    metadata = instance.metadata

    for i in range(len(instance)):
        token = Token()
        token[ID] = i + 1

        for field in fields:
            v = instance[field][i]
            if v is None:
                value = None
            elif isinstance(v, np.ndarray):
                value = [index[field][ch] for ch in v]
            else:
                value = index[field][v]
            if value is not None:
                token[field] = value

        if join:
            for f, ch in _CHARS_FIELDS_MAP.items():
                if ch in token:
                    value = join(ch, token[ch])
                    if value is not None:
                        token[f] = value

        tokens.append(token)
    
    return Sentence(tokens, metadata)

def shuffled_stream(instances, size=0):
    i = 0
    while True:
        random.shuffle(instances)
        for instance in instances:
            if size > 0 and i >= size:
                return
            i += 1
            yield instance
