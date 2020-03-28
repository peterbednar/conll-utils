from conllutils import FORM
from conllutils.distance import DEL, INS, SUB, TRN
from conllutils.distance import levenshtein_distance, tree_edit_distance

from conllutils.distance import _annotate

def sentence(s):
    return [{FORM: ch} for ch in s]

def test_levenshtein_distance():
    assert levenshtein_distance(sentence("abcabc"), sentence("abcabc")) == 0
    assert levenshtein_distance(sentence(""), sentence("")) == 0
    assert levenshtein_distance(sentence("abcabc"), sentence("")) == 6
    assert levenshtein_distance(sentence(""), sentence("abcabc")) == 6
    assert levenshtein_distance(sentence("abcabc"), sentence("bcab")) == 2
    assert levenshtein_distance(sentence("abcabc"), sentence("abccabc")) == 1
    assert levenshtein_distance(sentence("abccabc"), sentence("abcabc")) == 1
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbc")) == 2
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbc"), damerau=True) == 1

    assert levenshtein_distance(sentence("abcabc"), sentence("abcabc"), return_oprs=True)[1] == []
    assert levenshtein_distance(sentence(""), sentence(""), return_oprs=True)[1] == []
    assert levenshtein_distance(sentence("abcabc"), sentence(""), return_oprs=True)[1] == [(DEL, 0, -1), (DEL, 1, -1), (DEL, 2, -1), (DEL, 3, -1), (DEL, 4, -1), (DEL, 5, -1)]
    assert levenshtein_distance(sentence(""), sentence("abcabc"), return_oprs=True)[1] == [(INS, -1, 0), (INS, -1, 1), (INS, -1, 2), (INS, -1, 3), (INS, -1, 4), (INS, -1, 5)]
    assert levenshtein_distance(sentence("abcabc"), sentence("abccabc"), return_oprs=True)[1] == [(INS, -1, 3)]
    assert levenshtein_distance(sentence("abccabc"), sentence("abcabc"), return_oprs=True)[1] == [(DEL, 3, -1)]
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbca"), return_oprs=True)[1] == [(SUB, 2, 2), (SUB, 3, 3), (INS, -1, 6)]
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbca"), damerau=True, return_oprs=True)[1] == [(TRN, 2, 2), (INS, -1, 6)]

class _TestNode(object):

    def __init__(self, form):
        self.token = {FORM: form}
        self.children = []

def tree(s):
    return _parse(list(s), 0)[0]

def _parse(s, index):
    node = _TestNode(s[index])
    index += 1
    if s[index] == "(":
        index += 1
        while s[index] != ")":
            ch, index = _parse(s, index)
            node.children.append(ch)
        index += 1
    return node, index

def test_tree_edit_distance():
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(d))")) == 0
    assert tree_edit_distance(None, None) == 0
    assert tree_edit_distance(tree("a(bc(d))"), None) == 4
    assert tree_edit_distance(None, tree("a(bc(d))")) == 4
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(cb(d))")) == 2
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc)")) == 1
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(de))")) == 1

    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(d))"), return_oprs=True)[1] == []
    assert tree_edit_distance(None, None, return_oprs=True)[1] == []
    assert tree_edit_distance(tree("a(bc(d))"), None, return_oprs=True)[1] == [(DEL, 0, -1), (DEL, 1, -1), (DEL, 2, -1), (DEL, 3, -1)]
    assert tree_edit_distance(None, tree("a(bc(d))"), return_oprs=True)[1] == [(INS, -1, 0), (INS, -1, 1), (INS, -1, 2), (INS, -1, 3)]
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(cb(d))"), return_oprs=True)[1] == [(SUB, 0, 0), (SUB, 2, 2)]
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(de))"), return_oprs=True)[1] == [(INS, -1, 2)]

    nodes1, _, _ = _annotate(tree("a(bc(d))"))
    nodes2, _, _ = _annotate(tree("a(cb(d))"))
    assert nodes1[0].token[FORM] == "b"
    assert nodes2[0].token[FORM] == "c"
    assert nodes1[2].token[FORM] == "c"
    assert nodes2[2].token[FORM] == "b"

    nodes3, _, _ = _annotate(tree("a(bc(de))"))
    assert nodes3[2].token[FORM] == "e"
