import os
import re
import random
from collections import OrderedDict, Counter
from io import StringIO
import numpy as np

_EMPTY = "."
_MULTIWORD = "-"

FIELDS = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc",
          "form_norm", "lemma_norm", "form_chars", "lemma_chars", "form_norm_chars", "lemma_norm_chars", "upos_feats")

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, \
FORM_NORM, LEMMA_NORM, FORM_CHARS, LEMMA_CHARS, FORM_NORM_CHARS, LEMMA_NORM_CHARS, UPOS_FEATS = FIELDS

_BASE_FIELDS = (ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC)

_CHARS_FIELDS_MAP = {FORM: FORM_CHARS, LEMMA: LEMMA_CHARS, FORM_NORM: FORM_NORM_CHARS, LEMMA_NORM: LEMMA_NORM_CHARS}
_CHARS_FIELDS = set(_CHARS_FIELDS_MAP.values())

def empty_id(token_id, index):
    return (token_id, index, _EMPTY)

def multiword_id(start, end):
    return (start, end, _MULTIWORD)

class Token(dict):
    """A dictionary object representing a token in the sentence.

    A token can represent a regular syntactic word, or can be the multiword token spanning across multiple words (e.g.
    like in Spanish *vámonos* = *vamos nos*), or can be the empty token (inserted e.g. for analysis of ellipsis). Type
    of the token can be tested using the :py:attr:`is_multiword` and :py:attr:`is_empty` properties.

    A token can contain mappings for the following standard CoNLL-U fields:
        * ID: word index (integer starting from 1); or range of the indexes for multiword tokens; or decimal notation
          for empty tokens.
        * FORM: word form or punctuation symbol.
        * LEMMA: lemma or stem of word form.
        * UPOS: Universal part-of-speech tag.
        * XPOS: language-specific part-of-speech tag.
        * FEATS: list of morphological features from the universal feature inventory or language-specific extension.
        * HEAD: head of the current word in the dependency tree representation (ID or 0 for root).
        * DEPREL: Universal dependency relation to the HEAD.
        * DEPS: enhanced dependency graph in the form of head-deprel pairs.
        * MISC: any other annotation associated with the token. 

    CoNLLUtils package additionally defines the following extended fields:
        * UPOS_FEATS: concatenated POS and FEATS field (with added 'POS'=value pair to the FEATS list).
        * FORM_NORM, LEMMA_NORM: custom-normalized string form for FORM and LEMMA fields.
        * FORM_CHAR, LEMMA_CHAR, FORM_NORM_CHAR, LEMMA_NORM_CHAR: corresponding fields split to the list of characters.

    The ID values are parsed as the integers for regular words or tuples for multiword and empty tokens (see
    :py:func:`multiword_id` and :py:func:`empty_id` functions for more information).

    The HEAD values are parsed as the integers.

    The FORM, LEMMA, POS, XPOS, DEPREL, MISC, FORM_NORM and LEMMA_NORM values are strings.

    The FEATS or UPOS_FEATS values are strings or parsed as the dictionaries with attribute-value mappings and multiple
    values stored in the sets.

    The DEPS values are strings or parsed as the set of head-deprel tuples.

    """

    def __init__(self, fields={}):
        """Return an empty token or token with the fields initialized from the provided mapping object."""
        super().__init__(fields)

    @property
    def is_empty(self):
        """bool: True if the token is an *empty token*, otherwise False."""
        id = self.get(ID)
        return id[2] == _EMPTY if isinstance(id, tuple) else False

    @property
    def is_multiword(self):
        """bool: True if the token is a *multiword token*, otherwise False."""
        id = self.get(ID)
        return id[2] == _MULTIWORD if isinstance(id, tuple) else False

    def __repr__(self):
        return f"<{_id_to_str(self.get(ID))},{self.get(FORM)},{self.get(UPOS)}>"

    def copy(self):
        """Return a shallow copy of the token."""
        return Token(self)

class Sentence(list):

    def __init__(self, tokens=[], metadata=None):
        super().__init__(tokens)
        self.tokens = self
        self.metadata = metadata

    def get(self, id):
        if isinstance(id, str):
            id = _parse_id(id)
        start = id[0]-1 if isinstance(id, tuple) else id-1
        if start < 0:
            start = 0
        for token in self.tokens[start:]:
            if token[ID] == id:
                return token
        raise IndexError(f"token with ID {_id_to_str(id)} not found")

    def as_tree(self):
        return DependencyTree(self)

    def copy(self):
        return Sentence(self.tokens, self.metadata)

class Node(object):

    def __init__(self, token=None, parent=None):
        self.token = token
        self.parent = parent
        self.children = []

    @property
    def is_root(self):
        return self.parent == None

    @property
    def deprel(self):
        return self.token.get(DEPREL)

    def __repr__(self):
        return f"<{self.token},{self.deprel},{self.children}>"

class DependencyTree(object):

    def __init__(self, sentence):
        self.root = self._build(sentence)
        self.metadata = sentence.metadata

    def nodes(self, postorder=False):
        nodes = []
        self.visit(lambda n: nodes.append(n), postorder)
        return nodes

    def visit(self, f, postorder=False):
        if self.root:
            self._visit(self.root, f, postorder)
    
    @staticmethod
    def _visit(node, f, postorder):
        if not postorder:
            f(node)
        for ch in node.children:
            DependencyTree._visit(ch, f, postorder)
        if postorder:
            f(node)

    @staticmethod
    def _build(sentence):
        root = None
        tokens = sentence.tokens
        nodes = [Node() for _ in range(len(tokens))]

        for i, token in enumerate(tokens):
            id = token.get(ID)
            head = token.get(HEAD)

            if isinstance(id, tuple) or head is None:
                continue # skip empty and multiword tokens and tokens without HEAD
            index = id - 1 if id is not None else i

            nodes[index].token = token
            if head == 0:
                if root == None:
                    root = nodes[index]
                else:
                    raise ValueError("multiple roots")
            else:
                parent = nodes[head-1]
                nodes[index].parent = parent
                parent.children.append(nodes[index])

        return root

    def __repr__(self):
        return repr(self.root)

class Instance(dict):
    
    def __init__(self, length=0, fields={}, metadata=None):
        super().__init__(fields)
        self._length = length
        self.metadata = metadata

    def __len__(self):
        return self._length

    def token(self, index, fields=None):
        if fields == None:
            fields = self.keys()
        token = Instance(metadata=self.metadata)
        for f in fields:
            token[f] = self[f][index]
        return token

    @property
    def tokens(self):
        return [self.token(i) for i in range(self._length)]

    def as_tree(self):
        return DependencyTree(self)

    def copy(self):
        return Instance(self._length, self, self.metadata)

_NUM_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
NUM_NORM = u"__number__"

def normalize_lower(field, value):
    return value.lower()

def normalize_default(field, value):
    if _NUM_REGEX.match(value):
        return NUM_NORM
    return value.lower()

def split_default(field, value):
    if value == NUM_NORM:
        return None
    return tuple(value)

def _parse_sentence(lines, comments=[], skip_empty=True, skip_multiword=True,
                    parse_feats=False, parse_deps=False, upos_feats=True, normalize=normalize_default, split=split_default):
    sentence = Sentence()
    sentence.metadata = _parse_metadata(comments)

    for line in lines:
        token = _parse_token(line, parse_feats, parse_deps, upos_feats, normalize, split)
        if skip_empty and token.is_empty:
            continue
        if skip_multiword and token.is_multiword:
            continue
        sentence.append(token)

    return sentence

def _parse_metadata(comments):
    return [comment[1:].lstrip() for comment in comments]

def _parse_token(line, parse_feats=False, parse_deps=False, upos_feats=True, normalize=normalize_default, split=split_default):
    fields = line.split("\t")
    fields = {FIELDS[i] : fields[i] for i in range(min(len(fields), len(FIELDS)))}

    fields[ID] = _parse_id(fields[ID])

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
            if parse_feats:
                tag = _parse_feats(tag)
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

def _parse_id(s):
    if "." in s:
        token_id, index = s.split(".")
        return (int(token_id), int(index), _EMPTY)
    if "-" in s:
        start, end = s.split("-")
        return (int(start), int(end), _MULTIWORD)
    return int(s)

def _parse_feats(s):
    feats = OrderedDict()
    for key, value in [feat.split("=") for feat in s.split("|")]:
        if "," in value:
            value = set(value.split(","))
        feats[key] = value
    return feats

def _parse_deps(s):
    return set(map(lambda rel: (int(rel[0]), rel[1]), [rel.split(":") for rel in s.split("|")]))

def read_conllu(file, skip_empty=True, skip_multiword=True, parse_feats=False, parse_deps=False, upos_feats=True,
                normalize=normalize_default, split=split_default):
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
                    yield _parse_sentence(lines, comments, skip_empty, skip_multiword,
                            parse_feats, parse_deps, upos_feats,
                            normalize, split)
                    lines = []
                    comments = []
        if len(lines) > 0:
            yield _parse_sentence(lines, comments, skip_empty, skip_multiword,
                    parse_feats, parse_deps, upos_feats,
                    normalize, split)

def _token_to_str(token):
    return "\t".join([_field_to_str(token, field) for field in _BASE_FIELDS])

def _field_to_str(token, field):

    if field == ID:
        return _id_to_str(token[ID])

    if field not in token or token[field] is None:
        return "_"

    if field == FEATS:
        return _feats_to_str(token[FEATS])

    if field == DEPS:
        return _deps_to_str(token[DEPS])

    return str(token[field])

def _id_to_str(id):
    if isinstance(id, tuple):
        return f"{id[0]}.{id[1]}" if id[2] == _EMPTY else f"{id[0]}-{id[1]}"
    else:
        return str(id)

def _feats_to_str(feats):
    if isinstance(feats, str):
        return feats
    feats = [key + "=" + (",".join(sorted(value)) if isinstance(value, set) else value) for key, value in feats.items()]
    return "|".join(feats)        

def _deps_to_str(deps):
    if isinstance(deps, str):
        return deps
    deps = [f"{rel[0]}:{rel[1]}" for rel in sorted(deps, key=lambda rel: rel[0])]
    return "|".join(deps)

def write_conllu(file, data, write_metadata=True):
    if isinstance(data, Sentence):
        data = (data,)

    def _write_metadata(fp, metadata):
        if metadata:
            for comment in metadata:
                print("# " + comment, file=fp)

    def _write_tokens(fp, tokens):
        for token in tokens:
            print(_token_to_str(token), file=fp)

    if isinstance(file, str):
        file = open(file, "wt", encoding="utf-8")
    with file as fp:
        for sentence in data:
            if write_metadata:
                _write_metadata(fp, sentence.metadata)
            _write_tokens(fp, sentence.tokens)
            print(file=fp)

class _StringIO(StringIO):

    def close(self):
        pass

    def release(self):
        super().close()

def decode_conllu(s, skip_empty=True, skip_multiword=True, parse_feats=False, parse_deps=False, upos_feats=True,
                  normalize=normalize_default, split=split_default):
    return read_conllu(StringIO(s), skip_empty, skip_multiword, parse_feats, parse_deps, upos_feats, normalize, split)

def encode_conllu(data, encode_metadata=True):
    f = _StringIO()
    write_conllu(f, data, encode_metadata)
    s = f.getvalue()
    f.release()
    return s

def create_dictionary(sentences, fields={FORM, LEMMA, UPOS, XPOS, FEATS, DEPREL}):
    if ID in fields or HEAD in fields:
        raise ValueError("indexing ID or HEAD fields")

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
        min_fq = min_frequency[f] if isinstance(min_frequency, dict) else min_frequency
        for i, (s, fq) in enumerate(ordered):
            if fq >= min_fq:
                index[f][s] = i + 1
    return index

def create_inverse_index(index):
    return {f: {v: k for k, v in c.items()} for f, c in index.items()}

INDEX_FILENAME = "{0}index_{1}.txt"

_NONE_TOKEN = u"__none__"

def write_index(dirname, index, fields=None):
    if fields is None:
        fields = index.keys()
    index = create_inverse_index(index)
    for f in fields:
        filename = INDEX_FILENAME.format(dirname, f)
        with open(filename, "wt", encoding="utf-8") as fp:
            c = index[f]
            for i in range(1, len(c) + 1):
                token = c[i]
                if token is None:
                    token = _NONE_TOKEN
                print(token, file=fp)

def read_index(dirname, fields=None):
    if fields is None:
        fields = FIELDS
    index = {}
    for f in fields:
        filename = INDEX_FILENAME.format(dirname, f)
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
        fields = {HEAD} | set(index.keys())

    l = len(sentence)
    instance = Instance(l)
    instance.metadata = sentence.metadata

    for field in fields:
        array = np.full(l, None, dtype=np.object) if field in _CHARS_FIELDS else np.full(l, -1, dtype=np.int)

        for i, token in enumerate(sentence):
            value = token.get(field)
            if field == HEAD:
                if value is not None:
                    array[i] = value
            elif field in _CHARS_FIELDS:
                if value is not None:
                    chars = [index[field][ch] for ch in value]
                    value = np.array(chars, dtype=np.int)
                array[i] = value
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

    sentence = Sentence()
    sentence.metadata = instance.metadata

    for i in range(len(instance)):
        token = Token()
        token[ID] = i + 1

        for field in fields:
            vi = instance[field][i]
            if vi is None:
                continue
            if field == HEAD:
                value = vi
            elif field in _CHARS_FIELDS:
                value = tuple([index[field][ch] for ch in vi])
            else:
                value = index[field][vi]
            if value is not None:
                token[field] = value

        if join:
            for f, ch in _CHARS_FIELDS_MAP.items():
                if ch in token:
                    value = join(ch, token[ch])
                    if value is not None:
                        token[f] = value

        sentence.append(token)
    
    return sentence

def iterate_tokens(instances, fields=None):
    for instance in instances:
        for i in range(len(instance)):
            yield instance.token(i, fields)

def shuffled_stream(instances, total_size=None, batch_size=None, random=random):
    i = 0
    batch = []
    instances = list(instances)
    if not instances:
        return
    while True:
        random.shuffle(instances)
        for instance in instances:
            if total_size is not None and i >= total_size:
                return
            i += 1
            if batch_size is not None:
                batch.append(instance)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            else:
                yield instance
