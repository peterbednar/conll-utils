from conllutils import FORM
from conllutils.distance import DEL, INS, SUB, TRN
from conllutils.distance import levenshtein_distance, tree_edit_distance, dict_edit_distance, k_nearest_neighbors

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

    assert levenshtein_distance(sentence("abcabc"), sentence("abcabc"), normalize=True) == 0
    assert levenshtein_distance(sentence(""), sentence(""), normalize=True) == 0
    assert levenshtein_distance(sentence("abcabc"), sentence(""), normalize=True) == 1
    assert levenshtein_distance(sentence(""), sentence("abcabc"), normalize=True) == 1
    assert levenshtein_distance(sentence("abcabc"), sentence("abccabc"), normalize=True) == 2*1/(6 + 7 + 1)
    assert levenshtein_distance(sentence("abccabc"), sentence("abcabc"), normalize=True) == 2*1/(7 + 6 + 1)
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbca"), normalize=True) == 2*3/(6 + 7 + 3)
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbca"), damerau=True, normalize=True) == 2*2/(6 + 7 + 2)

    assert levenshtein_distance(sentence("abcabc"), sentence("abcabc"), return_oprs=True)[1] == []
    assert levenshtein_distance(sentence(""), sentence(""), return_oprs=True)[1] == []
    assert levenshtein_distance(sentence("abcabc"), sentence(""), return_oprs=True)[1] == [(DEL, 0), (DEL, 1), (DEL, 2), (DEL, 3), (DEL, 4), (DEL, 5)]
    assert levenshtein_distance(sentence(""), sentence("abcabc"), return_oprs=True)[1] == [(INS, 0), (INS, 1), (INS, 2), (INS, 3), (INS, 4), (INS, 5)]
    assert levenshtein_distance(sentence("abcabc"), sentence("abccabc"), return_oprs=True)[1] == [(INS, 3)]
    assert levenshtein_distance(sentence("abccabc"), sentence("abcabc"), return_oprs=True)[1] == [(DEL, 3)]
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbca"), return_oprs=True)[1] == [(SUB, 2, 2), (SUB, 3, 3), (INS, 6)]
    assert levenshtein_distance(sentence("abcabc"), sentence("abacbca"), damerau=True, return_oprs=True)[1] == [(TRN, 2, 2), (INS, 6)]

class _TestNode(object):

    def __init__(self, form, index):
        self.index = index
        self.token = {FORM: form}
        self._children = []

    def __iter__(self):
        return iter(self._children)

def tree(s):
    return _parse(list(s), 0)[0]

def _parse(s, index):
    node = _TestNode(s[index], index)
    index += 1
    if s[index] == "(":
        index += 1
        while s[index] != ")":
            ch, index = _parse(s, index)
            node._children.append(ch)
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

    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(d))"), normalize=True) == 0
    assert tree_edit_distance(None, None, normalize=True) == 0
    assert tree_edit_distance(tree("a(bc(d))"), None, normalize=True) == 1
    assert tree_edit_distance(None, tree("a(bc(d))"), normalize=True) == 1
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(cb(d))"), normalize=True) == 2*2/(4 + 4 + 2)
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc)"), normalize=True) == 2*1/(4 + 3 + 1)
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(de))"), normalize=True) == 2*1/(4 + 5 + 1)

    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(d))"), return_oprs=True)[1] == []
    assert tree_edit_distance(None, None, return_oprs=True)[1] == []
    assert tree_edit_distance(tree("a(bc(d))"), None, return_oprs=True)[1] == [(DEL, 2), (DEL, 5), (DEL, 3), (DEL, 0)]
    assert tree_edit_distance(None, tree("a(bc(d))"), return_oprs=True)[1] == [(INS, 2), (INS, 5), (INS, 3), (INS, 0)]
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(cb(d))"), return_oprs=True)[1] == [(SUB, 2, 2), (SUB, 3, 3)]
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(de))"), return_oprs=True)[1] == [(INS, 6)]

    nodes1, _, _ = _annotate(tree("a(bc(d))"))
    nodes2, _, _ = _annotate(tree("a(cb(d))"))
    assert nodes1[0].token[FORM] == "b"
    assert nodes2[0].token[FORM] == "c"
    assert nodes1[2].token[FORM] == "c"
    assert nodes2[2].token[FORM] == "b"

    nodes3, _, _ = _annotate(tree("a(bc(de))"))
    assert nodes3[2].token[FORM] == "e"

def test_dict_edit_distance():
    assert dict_edit_distance({}, {}) == 0
    assert dict_edit_distance("a=a|b=b|c=c", "a=a|b=b|c=c") == 0
    assert dict_edit_distance("a=a|b=b|c=c", {}) == 3
    assert dict_edit_distance({}, "a=a|b=b|c=c") == 3
    assert dict_edit_distance("a=a|b=b|c=c", "b=b") == 2
    assert dict_edit_distance("a=a|b=b|c=c", "a=b|b=a") == 3

    assert dict_edit_distance({}, {}, normalize=True) == 0
    assert dict_edit_distance("a=a|b=b|c=c", "a=a|b=b|c=c", normalize=True) == 0
    assert dict_edit_distance("a=a|b=b|c=c", {}, normalize=True) == 1
    assert dict_edit_distance({}, "a=a|b=b|c=c", normalize=True) == 1
    assert dict_edit_distance("a=a|b=b|c=c", "b=b", normalize=True) == 2/3
    assert dict_edit_distance("a=a|b=b|c=c", "a=b|b=a", normalize=True) == 3/3

    assert dict_edit_distance({}, {}, return_oprs=True)[1] == set()
    assert dict_edit_distance("a=a|b=b|c=c", "a=a|b=b|c=c", return_oprs=True)[1] == set()
    assert dict_edit_distance("a=a|b=b|c=c", {}, return_oprs=True)[1] == {(DEL, "a"), (DEL, "b"), (DEL, "c")}
    assert dict_edit_distance({}, "a=a|b=b|c=c", return_oprs=True)[1] == {(INS, "a"), (INS, "b"), (INS, "c")}
    assert dict_edit_distance("a=a|b=b|c=c", "b=b", return_oprs=True)[1] == {(DEL, "a"), (DEL, "c")}
    assert dict_edit_distance("a=a|b=b|c=c", "a=b|b=a", return_oprs=True)[1] == {(SUB, "a"), (SUB, "b"), (DEL, "c")}

def test_k_nearest_neighbours():
    assert k_nearest_neighbors(sentence("a"), [sentence(s) for s in ["abcd", "abc", "ab", "a", "ba", "bac"]], k=3) == [(3, 0), (2, 1), (4, 1)]
    assert k_nearest_neighbors(sentence("a"), [sentence(s) for s in ["abcd", "abc", "ab", "a", "ba", "bac"]], k=3, return_distance=False) == [3, 2, 4]
