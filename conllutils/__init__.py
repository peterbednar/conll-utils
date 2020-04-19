import os
import re
from collections import Counter
from collections.abc import MutableMapping
from operator import itemgetter
from io import StringIO
import numpy as np

_EMPTY = 0
_MULTIWORD = 1

FIELDS = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = FIELDS
_FIELD_SET = set(FIELDS)

def empty_id(word_id, index=1):
    """Return new ID value for empty token indexed by `word_id` starting from 0 and `index` starting from 1.

    The empty token ID is encoded as a tuple with id[0] = `word_id` and id[1] = `index`. For more information about the
    ordering of the empty tokens in the sentence, see :class:`Sentence` class.

    Raises:
        ValueError: If `word_id` < 0 or `index` < 1.
    """
    if word_id < 0 or index < 1:
        raise ValueError("word_id must be >= 0 and index >= 1.")
    return (word_id, index, _EMPTY)

def multiword_id(start, end):
    """Return new ID value for multiword token spanning in the sentence across the words with ID from `start` to `end`
    (inclusive).

    The multiword token ID is encoded as a tuple with id[0] = `start` and id[1] = `end`. For more information about the
    ordering of the multiword tokens in the sentence, see :class:`Sentence` class.

    Raises:
        ValueError: If `start` < 1 or `end` <= `start`.
    """

    if start < 1 or end <= start:
        raise ValueError("start must be >= 1 and end > start.")
    return (start, end, _MULTIWORD)

class Token(dict):
    """A dictionary type representing a token in the sentence.

    A token can represent a regular *syntactic word*, or can be a *multiword token* spanning across multiple words
    (e.g. like in Spanish *vámonos* = *vamos nos*), or can be an *empty token* (inserted in the extended dependency
    tree, e.g. for the analysis of ellipsis). Type of the token can be tested using the read-only :attr:`is_multiword`
    and :attr:`is_empty` properties.

    A token can contain mappings for the following standard CoNLL-U fields:
        * ID: word index (integer starting from 1); or range of the indexes for multiword tokens; or decimal notation
          for empty tokens.
        * FORM: word form or punctuation symbol.
        * LEMMA: lemma or stem of word form.
        * UPOS: Universal part-of-speech tag.
        * XPOS: language-specific part-of-speech tag.
        * FEATS: list of morphological features from the Universal feature inventory or language-specific extension.
        * HEAD: head of the current word in the dependency tree representation (ID or 0 for root).
        * DEPREL: Universal dependency relation to the HEAD.
        * DEPS: enhanced dependency graph in the form of head-deprel pairs.
        * MISC: any other annotation associated with the token. 

    The ID values are parsed as the integers for regular words or tuples for multiword and empty tokens (see
    :func:`multiword_id` and :func:`empty_id` functions for more information).

    The HEAD values are parsed as the integers.

    The FORM, LEMMA, POS, XPOS, DEPREL and MISC values are strings.

    The FEATS are strings or parsed as the dictionaries with attribute-value mappings and multiple values stored in the
    sets.

    The DEPS values are strings or parsed as the set of head-deprel tuples.

    """

    def __init__(self, fields=(), **kwargs):
        """Create an empty token or token with the fields initialized from the provided mapping object or keyword
        arguments."""
        super().__init__(fields, **kwargs)

    @property
    def is_empty(self):
        """bool: True if the token is an empty token, otherwise False."""
        id = self.get(ID)
        return id[2] == _EMPTY if isinstance(id, tuple) else False

    @property
    def is_multiword(self):
        """bool: True if the token is a multiword token, otherwise False."""
        id = self.get(ID)
        return id[2] == _MULTIWORD if isinstance(id, tuple) else False

    def to_collu(self):
        """Return a string representation of the token in the CoNLL-U format."""
        return _token_to_str(self)

    def __getattr__(self, name):
        if name in _FIELD_SET:
            return self[name]
        else:
            raise AttributeError(f"Token has no attribute {name}.")

    def __setattr__(self, name, value):
        if name in _FIELD_SET:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _FIELD_SET:
            del self[name]
        else:
            super().__delattr__(name)

    def __repr__(self):
        return f"<{_id_to_str(self.get(ID))},{self.get(FORM)},{self.get(UPOS)}>"

    def copy(self):
        """Return a shallow copy of the token."""
        return Token(self)

class Sentence(list):
    """A list type representing the sentence, i.e. the sequence of tokens.

    For valid CoNLL-U sentences, tokens have to be ordered according to their IDs. The syntactic words form the sequence
    with ID=1, 2, 3, etc. Multiword tokens with the range ID 'start-end' are inserted before the first word in the range
    (i.e. before the word with ID=start). The ranges of all multiword tokens must be non-empty and non-overlapping.
    Empty tokens with the decimal IDs 'token_id.index' are inserted in the index order at the beginning of the sentence
    (if token_id=0), or immediately after the word with ID=token_id.

    Note that the Sentence methods are not checking the order of the tokens, and it is up to the programmer to preserve
    the correct ordering.

    The Sentence class provides :meth:`words` method to extract only the sequence of syntactic words without the empty
    or multiword tokens, and :meth:`raw_tokens` method to extract the sequence of raw tokens (i.e. how the sentence is
    written orthographically with the multiword tokens).

    For example, for the Spanish sentence::
    
        1-2     vámonos
        1       vamos
        2       nos
        3-4     al
        3       a
        4       el
        5       mar

    the :meth:`words` method returns the sequence of expanded syntactic words 'vamos', 'nos', 'a', 'el', 'mar', and the
    :meth:`raw_tokens` returns sequence for raw text 'vámonos', 'al', 'mar'.
    
    For the sentence with empty tokens::
    
        1      Sue
        2      likes
        3      coffee
        4      and
        5      Bill
        5.1    likes
        6      tea

    both :meth:`words` and :meth:`raw_tokens` methods return the sequence without the empty tokens 'Sue', 'likes',
    'coffee', 'and', 'Bill', 'tea'.

    Attributes:
        metadata (any): Any optional data associated with the sentence. By default for the CoNLL-U format, `metadata`
            are parsed as the list of strings (without the trailing '#') from the comment lines before the sentence.

    """

    def __init__(self, tokens=(), metadata=None):
        """Create an empty sentence or initialize a new sentence with the `tokens` from the provided sequence and
        optional `metadata`.
        """
        super().__init__(tokens)
        self.metadata = metadata

    def is_projective(self, return_arcs=False):
        """Return True if this sentence can be represented as the projective dependency tree, otherwise False.

        See :meth:`DependencyTree.is_projective` method for more information.
        """
        return _is_projective([token[HEAD] for token in self.words()], return_arcs)

    def get(self, id):
        """Return token with the specified ID.
        
        The `id` argument can be an integer from 1, tuple generated by :func:`empty_id` or :func:`multiword_id`
        functions, or string in CoNLL-U notation (e.g. "1" for words, "2-3" for multiword tokens, or "0.1" for empty
        tokens). Note that the implementation assumes the proper ordering of the tokens according to their IDs.

        Raises:
            IndexError: If a token with the `id` cannot be found in the sentence.
        """
        if isinstance(id, str):
            id = _parse_id(id)
        start = id[0]-1 if isinstance(id, tuple) else id-1
        if start < 0:
            start = 0
        for token in self[start:]:
            if token[ID] == id:
                return token
        raise IndexError(f"Token with ID {_id_to_str(id)} not found.")

    def tokens(self):
        """Return an iterator over all tokens in the sentence (alias to ``iter(self)``)."""
        return iter(self)

    def raw_tokens(self):
        """Return an iterator over all raw tokens representing the written text of the sentence.

        The raw tokens are all multiword tokens and all words outside of the multiword ranges (excluding the empty
        tokens). Note that the implementation assumes the proper ordering of the tokens according to their IDs.
        """
        index = 1
        prev_end = 0
        for token in self:
            if token.is_empty:
                continue
            if token.is_multiword:
                prev_end = token[ID][1]
                yield token
            else:
                if index > prev_end:
                    yield token
                index += 1

    def words(self):
        """Return an iterator over all syntactic words (i.e. without multiword and empty tokens)."""
        for token in self:
            if not (token.is_empty or token.is_multiword):
                yield token

    def to_tree(self):
        """Return a dependency tree representation of the sentence.
 
        See :class:`DependencyTree` class for more information. Note that the implementation assumes the proper ordering
        of the tokens according to their IDs.

        Raises:
            ValueError: If the sentence contains the words without the HEAD field, or when the sentence does not have
                exactly one root with HEAD = 0.
        """
        return DependencyTree(self)

    def to_instance(self, index, fields=None):
        """Return an instance representation of the sentence with the values indexed by the `index`.

        Optional `fields` argument specifies a subset of the fields added into the instance. By default, HEAD field and
        all fields from the `index` are included. See :class:`Instance` class for more information.

        Raises:
            KeyError: If some of the `fields` are not indexed in the `index`.
        """
        return _map_to_instance(self, index, fields)

    def to_conllu(self, metadata=True):
        """Return a string representation of the sentence in the CoNLL-U format.

        If the `metadata` argument is True (default), the string also includes comments generated from the metadata.
        """
        return _sentence_to_str(self, metadata)

    @staticmethod
    def from_conllu(s, multiple=False, **kwargs):
        """Parse a sentence (or list of sentences) from the string in the CoNLL-U format.
        
        If the argument `multiple` is True, the function returns the list of all sentences parsed from the string.
        Otherwise (default), it returns only the first sentence. This function supports all additional keyword arguments
        as the :func:`read_conllu` function.

        Raises:
	        ValueError: If there is an error parsing at least one sentence from the string.
        """
        itrs = read_conllu(StringIO(s), **kwargs)
        result = list(itrs) if multiple else next(itrs, None)
        if not result:
            raise ValueError("No sentence found.")
        return result

    def copy(self):
        """Return a shallow copy of the sentence."""
        return Sentence(self, self.metadata)

def _parse_sentence(lines, comments, parse_feats, parse_deps):
    sentence = Sentence()
    sentence.metadata = _parse_metadata(comments)

    for line in lines:
        token = _parse_token(line, parse_feats, parse_deps)
        sentence.append(token)

    return sentence

def _parse_metadata(comments):
    return [comment[1:].lstrip() for comment in comments]

def _parse_token(line, parse_feats, parse_deps):
    fields = line.split("\t")
    fields = {FIELDS[i] : fields[i] for i in range(min(len(fields), len(FIELDS)))}

    fields[ID] = _parse_id(fields[ID])

    for f in (FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC):
        if f in fields and fields[f] == "_":
            del(fields[f])

    if parse_feats and FEATS in fields:
        fields[FEATS] = _parse_feats(fields[FEATS])

    if HEAD in fields:
        fields[HEAD] = int(fields[HEAD])

    if parse_deps and DEPS in fields:
        fields[DEPS] = _parse_deps(fields[DEPS])

    return Token(fields)

def _parse_id(s):
    if "." in s:
        word_id, index = s.split(".")
        return empty_id(int(word_id), int(index))
    if "-" in s:
        start, end = s.split("-")
        return multiword_id(int(start), int(end))
    return int(s)

def _parse_feats(s):
    feats = {}
    for key, value in [feat.split("=") for feat in s.split("|")]:
        if "," in value:
            value = set(value.split(","))
        feats[key] = value
    return feats

def _parse_deps(s):
    return set(map(lambda rel: (int(rel[0]), rel[1]), [rel.split(":") for rel in s.split("|")]))

def _sentence_to_str(sentence, encode_metadata):
    lines = []
    if encode_metadata:
        lines = _metadata_to_str(sentence.metadata)
    lines += [_token_to_str(token) for token in sentence]
    return "\n".join(lines)

def _metadata_to_str(metadata):
    if isinstance(metadata, list):
        return ["# " + str(comment) for comment in metadata]
    else:
        return []

def _token_to_str(token):
    return "\t".join([_field_to_str(token, field) for field in FIELDS])

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

class Node(object):
    """A node in the dependency tree corresponding to the syntactic word in the sentence.

    A node object is iterable, and returns an iterator over the direct children. ``len(node)`` returns the number of
    children, and ``node[i]`` returns the `i`-th child (or sublist of children, if `i` is the slice of indices).

    Attributes:
        index (int): The index of the word in the sentence (from 0).
        token (:class:`Token` or indexed token view): The corresponding syntactic word.
        parent (:class:`Node`): The parent (HEAD) of the node, or `None` for the root. 

    """
    def __init__(self, index, token):
        self.index = index
        self.token = token
        self.parent = None
        self._children = []

    @property
    def is_root(self):
        """bool: True, if the node is the root of the tree (has no parent)."""
        return self.parent == None

    @property
    def is_leaf(self):
        """bool: True, if the node is a leaf node (has no children)."""
        return len(self) == 0

    @property
    def deprel(self):
        """str or int: Universal dependency relation to the HEAD stored in the ``token[DEPREL]``, or `None` if the token
        does not have DEPREL field."""
        return self.token.get(DEPREL)

    def __getitem__(self, i):
        # Return `i`-th child of the node or sublist of children, if `i` is the slice of indices.
        return self._children[i]

    def __len__(self):
        # Return the number of children.
        return len(self._children)

    def __iter__(self):
        # Return an iterator over the children.
        return iter(self._children)

    def __repr__(self):
        return f"<{self.token},{self.deprel},{self._children}>"

class DependencyTree(object):
    """A dependency tree representation of the sentence.

    A *basic* dependency tree is a labeled tree structure where each node of the tree corresponds to exactly one
    syntactic word in the sentence. The relations between the node and its parent (head) are labeled with the Universal
    dependencies and stored in the HEAD and DEPREL fields of the corresponding word.

    The DependencyTree class should not be instantiated directly. Use the :meth:`Sentence.to_tree` or
    :meth:`Instance.to_tree` methods to create a dependency representation for the sentence or indexed instance. The
    implementation of nodes is provided by :class:`Node` class.

    The dependency tree object is iterable and returns an iterator over all nodes in the order of corresponding words in
    the sentence. ``len(tree)`` returns the number of nodes.

    Note that the dependency tree is constructed only from the basic dependency relations. Enhanced dependency relations
    stored in the DEPS field are not included in the tree.

    Attributes:
        root (:class:`Node`): The root of the tree. 
        nodes (list of :class:`Node`): The list of all nodes in the sentence order.
        metadata (any): Any optional data associated with the tree, by default copied from the sentence or indexed
            instance.

    """
    def __init__(self, sentence):
        self.root, self.nodes = self._build(sentence)
        self.metadata = sentence.metadata

    def __len__(self):
        # Return the number of nodes.
        return len(self.nodes)

    def __iter__(self):
        # Return an iterator over all nodes in the sentence order.
        return iter(self.nodes)

    def is_projective(self, return_arcs=False):
        """Return True if the dependency tree is projective, otherwise False.

        A dependency tree is projective when all its arcs are projective, i.e. for all arcs (`i`, `j`) from parent `i`
        to child `j` and for all nodes `k` between the `i` and `j` in the sentence, there must be a path from `i` to
        `k`.

        If the argument `return_arcs` is True, the function returns the list of conflicting non-projective arcs. For
        projective trees the list is empty.

        """
        return _is_projective([node.token[HEAD] for node in self.nodes], return_arcs)

    def leaves(self):
        """Return an iterator over all leaves of the tree in the sentence order."""
        for node in self:
            if node.is_leaf:
                yield node

    def inorder(self):
        """Return an iterator traversing in-order over all nodes."""
        return self._traverse(self.root, inorder=True)

    def preorder(self):
        """Return an iterator traversing pre-order over all nodes."""
        return self._traverse(self.root, preorder=True)

    def postorder(self):
        """Return an iterator traversing post-order over all nodes."""
        return self._traverse(self.root, postorder=True)

    def __repr__(self):
        return repr(self.root)

    @staticmethod
    def _traverse(node, inorder=False, preorder=False, postorder=False):
        if node is None:
            return

        consumed = False
        if preorder:
            consumed = True # consume preorder
            yield node

        for child in node:
            if inorder and not consumed and node.index < child.index:
                consumed = True # consume inorder
                yield node
            yield from DependencyTree._traverse(child, inorder, preorder, postorder)

        if postorder or not consumed: # for postorder or right-most inorder
            yield node

    @staticmethod
    def _build(sentence):
        if isinstance(sentence, Instance):
            tokens = sentence.tokens()
        else:
            tokens = sentence.words() # only the syntactic words

        nodes = [Node(i, token) for i, token in enumerate(tokens)]
        if not nodes:
            return None, []

        root = None
        for index, node in enumerate(nodes):
            # token can be syntactic word or indexed token view
            token = node.token
            head = token.get(HEAD)
            if head is None or head == -1:
                raise ValueError(f"Token at {index} is without HEAD.")

            if head == 0:
                if root == None:
                    root = node
                else:
                    raise ValueError(f"Multiple roots found at {index}.")
            else:
                parent = nodes[head-1]
                node.parent = parent
                parent._children.append(node)

        if root is None:
            raise ValueError("No root found.")

        return root, nodes

class _IndexedToken(MutableMapping):
    """A mutable mapping view representing `i`-th token of the indexed instance."""
    def __init__(self, index, fields):
        self._index = index
        self._fields = fields

    def __len__(self):
        # Return the number of mapped fields.
        return len(self._fields)
    
    def __iter__(self):
        # Return an iterator over the mapped fields.
        return iter(self._fields)

    def __getitem__(self, field):
        # Return the value of the `field`. Raises a KeyError if the `field` is not mapped in the instance.
        return self._fields[field][self._index]

    def __setitem__(self, field, value):
        # Set the value of the `field`. Raises a KeyError if the `field` is not mapped in the instance.
        if not field in self._fields:
            raise KeyError(field)
        self._fields[field][self._index] = value

    def __delitem__(self, key):
        # Remove key operation is not supported for the token view.
        raise TypeError("Not supported for token views.")

class Instance(dict):
    """An indexed representation of the sentence in the compact numerical form.

    An instance can be created from a sentence using the :meth:`Sentence.to_instance` method. The sentence values are 
    mapped to the numerical indexes by the provided *index* mapping. The index for a set of sentences can be created
    with the :func:`create_index` function.

    An instance is a dictionary type where each field is mapped to the NumPy array with the integer values continuously
    indexed for all tokens in the sentence, i.e. the field value of the `i`-th token is stored as ``instance[field][i]``.
    The length of all mapped arrays is equal to the length of the sentence.

    The ID field is not stored in the instance. Note that this also means that the type of tokens is not preserved.

    The HEAD field and string-valued fields (i.e. FORM, LEMMA, UPOS, XPOS, DEPREL, MISC, FORM_NORM and LEMMA_NORM) are
    indexed and stored in the ``numpy.int`` array. The FEATS and DEPS are indexed as unparsed strings, i.e. the features
    or dependencies are not indexed separately.

    By default, unknown values (i.e. values not mapped in the provided index) are stored as 0. Missing values (i.e. when
    some token does not have indexed field) are stored as -1 or as `None` for the character fields.

    Attributes:
        metadata (any): Any optional data associated with the instance, by default copied from the sentence.

    """
    def __init__(self, fields=(), metadata=None):
        super().__init__(fields)
        self.metadata = metadata

    @property
    def length(self):
        """int: The length of the intance (i.e. the number of tokens in the indexed sentence)."""
        for data in self.values():
            return len(data)
        return 0

    def __getattr__(self, name):
        if name in _FIELD_SET:
            return self[name]
        else:
            raise AttributeError(f"Instance has no attribute {name}.")

    def __setattr__(self, name, value):
        if name in _FIELD_SET:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _FIELD_SET:
            del self[name]
        else:
            super().__delattr__(name)

    def is_projective(self, return_arcs=False):
        """Return True if this instance can be represented as the projective dependency tree, otherwise False.

        See :meth:`DependencyTree.is_projective` method for more information.
        """
        return _is_projective(self[HEAD], return_arcs)

    def token(self, i):
        """Return a view to the `i`-th token of the instance.

        The view is a mutable mapping object, which maps fields to the scalar values stored in the instance at the
        `i`-th position, i.e. for the values of the `i`-th token view, the following condition holds
        ``token[field] == instance[field][i]``.

        The view object supports all mapping methods and operations except the deleting of the field or setting the
        value of the field not indexed in the instance.
        """
        return _IndexedToken(i, self)

    def tokens(self):
        """Return an iterator over all tokens. The iterated values are token view objects."""
        for i in range(self.length):
            yield self.token(i)

    def to_tree(self):
        """Return a dependency tree representation of the instance.
 
        See :class:`DependencyTree` class for more information. All tokens referenced in the tree are indexed views, as
        it is described for the :meth:`token` method. Note that the implementation assumes proper ordering of the tokens
        and that the instance does not contain empty or multiword tokens.

        Raises:
            ValueError: If the instance contains the tokens without the HEAD field (HEAD = -1), or when the instance
                does not have exactly one root with HEAD = 0.
        """
        return DependencyTree(self)
    
    def to_sentence(self, inverse_index, fields=None):
        """Return a new sentence build from the instance with the values re-indexed by the `inverse_index`.

        Optional `fields` argument specifies a subset of the fields added into the sentence. By default, all instance
        fields are included. The ID values are always generated as the sequence of integers starting from 1, which
        corresponds to the sequence of lexical words without the empty or multiword tokens.

        This operation is inverse to the indexing in :meth:`Sentence.to_instance` method.

        Raises:
            KeyError: If some of the instance values is not mapped in the `inverse_index`.
        """
        return _map_to_sentence(self, inverse_index, fields)

    def copy(self):
        """Return a shallow copy of the instance."""
        return Instance(self, self.metadata)

def _is_projective(heads, return_arcs=False):

    if return_arcs:
        arcs = []

    for i in range(len(heads)):
        if heads[i] == None or heads[i] < 0:
            continue

        for j in range(i + 1, len(heads)):
            if heads[j] == None or heads[j] < 0:
                continue

            arc1_0 = min(i + 1, heads[i])
            arc1_1 = max(i + 1, heads[i])
            arc2_0 = min(j + 1, heads[j])
            arc2_1 = max(j + 1, heads[j])

            if ((arc1_0 == arc2_0 and arc1_1 == arc2_1) or # cycle
                (arc1_0 < arc2_0 and arc2_0 < arc1_1 and arc1_1 < arc2_1) or # crossing
                (arc2_0 < arc1_0 and arc1_0 < arc2_1 and arc2_1 < arc1_1)):  # crossing
                if return_arcs:
                    arcs.append((i, j))
                else:
                    return False

    if return_arcs:
        return arcs
    else:
        return True

def read_conllu(file, parse_feats=False, parse_deps=False):
    """Read the CoNLL-U file and return an iterator over the parsed sentences.

    The `file` argument can be a path-like or file-like object.

    To parse values of FEATS or DEPS fields to dictionaries or sets of tuples, set the `parse_feats` or `parse_deps`
    arguments to True. By default the features and dependencies are not parsed and values are stored as a string.
    """
    if isinstance(file, (str, os.PathLike)):
        file = open(file, "rt", encoding="utf-8")

    with file:
        lines = []
        comments = []

        for line in file:
            line = line.rstrip("\r\n")
            if line.startswith("#"):
                comments.append(line)
            elif line.lstrip():
                lines.append(line)
            else :
                if len(lines) > 0:
                    yield _parse_sentence(lines, comments,
                        parse_feats, parse_deps)
                    lines = []
                    comments = []

        if len(lines) > 0:
            yield _parse_sentence(lines, comments,
                parse_feats, parse_deps)

def write_conllu(file, data, write_metadata=True):
    """Write the sentences to the CoNLL-U file.

     The `file` argument can be a path-like or file-like object. Written `data` is an iterable object of sentences or
     one sentence. If the `write_metadata` argument is True (default), sentence metadata are encoded as the comments and
     written to the file.
    """
    if isinstance(data, Sentence):
        data = (data,)

    if isinstance(file, (str, os.PathLike)):
        file = open(file, "wt", encoding="utf-8")

    with file as fp:
        for sentence in data:
            if write_metadata and sentence.metadata:
                for comment in _metadata_to_str(sentence.metadata):
                    print(comment, file=fp)
            for token in sentence:
                print(_token_to_str(token), file=fp)
            print(file=fp)

def _create_dictionary(sentences, fields):
    if ID in fields or HEAD in fields:
        raise ValueError("Indexing ID or HEAD fields.")

    dic = {f: Counter() for f in fields}
    for sentence in sentences:
        for token in sentence:
            for f in fields:
                s = token.get(f)
                if s is None:
                    continue
                if isinstance(s, (list, tuple)):
                    for ch in s:
                        dic[f][ch] += 1
                else:
                    dic[f][s] += 1
    return dic

def create_index(sentences, fields=None, min_frequency=1):
    """Return an index mapping the string values of the `sentences` to integer indexes.

    An index is a nested dictionary where the indexes for the field values are stored as ``index[field][value]``. See
    :meth:`Sentence.to_instance` method for usage of the index dictionary for sentence indexing.

    For each field, the indexes are assigned to the string values starting from 1 according to their descending
    frequency of occurrences in the sentences, i.e. the most frequent value has index 1, second one index 2, etc.
    Index 0 represents an *unknown* value, and the dictionary returns 0 for all unmapped values.

    For mapping of instances to the sentences, use :func:`create_inverse_index` function to create an inverse mapping
    from the indexes to the string values.

    Args:
        sentences (iterable): The indexed sentences.
        fields (set): The set of indexed fields included in the index.
        min_frequency (int or dictionary): If specified, the field values with a frequency lower than `min_frequency`
            are discarded from the index. By default, all values are preserved. The `min_frequency` can be specified as
            an integer for all fields, or as a dictionary setting the frequency for the specific field.

    Raises:
        ValueError: If the indexed fields include ID or HEAD field.
    """
    if fields is None:
        fields = {FORM, LEMMA, UPOS, XPOS, FEATS, DEPREL}

    dic = _create_dictionary(sentences, fields)
    index = {f: Counter() for f in dic.keys()}

    for f, c in dic.items():
        ordered = sorted(c.items(), key=itemgetter(1,0), reverse=True)
        min_fq = min_frequency.get(f, 1) if isinstance(min_frequency, dict) else min_frequency
        for i, (s, fq) in enumerate(ordered):
            if fq >= min_fq:
                index[f][s] = i + 1
            else:
                break

    return index

def create_inverse_index(index):
    """Return an inverse index mapping the integer indexes to string values.

    For the `index` with mapping index[field][v] = i, the inverse index has mapping inverse_index[field][i] = v. See
    :meth:`Instance.to_sentence` method for usage of the inverse index for transformation of instances to sentences.
    """
    return {f: {v: k for k, v in c.items()} for f, c in index.items()}

def _map_to_instance(sentence, index, fields=None):
    if fields is None:
        fields = {HEAD} | set(index.keys())

    length = len(sentence)
    instance = Instance()
    instance.metadata = sentence.metadata

    for field in fields:
        array = np.full(length, -1, dtype=np.int)
        for i, token in enumerate(sentence):
            value = token.get(field)
            if field == HEAD:
                array[i] = value if value is not None else -1
            else:
                array[i] = index[field][value]

        instance[field] = array
    
    return instance

def _map_to_sentence(instance, inverse_index, fields=None):
    if fields is None:
        fields = instance.keys()

    sentence = Sentence()
    sentence.metadata = instance.metadata

    for i in range(instance.length):
        token = Token()
        token[ID] = i + 1

        for field in fields:
            vi = instance[field][i]
            if field == HEAD:
                value = vi if vi != -1 else None
            else:
                value = inverse_index[field].get(vi)
            if value is not None:
                token[field] = value

        sentence.append(token)
    
    return sentence
