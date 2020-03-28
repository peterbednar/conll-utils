from conllutils.distance import DEL, INS, SUB, TRN
from conllutils.distance import levenshtein_distance, tree_edit_distance

def token_edit_distance(t1, t2, opr):
    if opr == DEL:
        return 1    # insertion
    if opr == INS:
        return 1    # deletion
    if opr == TRN:
        return 1    # transposition
    return 0 if t1 == t2 else 1  # substitution

def test_levenshtein_distance():
    assert levenshtein_distance("abcabc", "abcabc", cost=token_edit_distance) == 0
    assert levenshtein_distance("", "", cost=token_edit_distance) == 0
    assert levenshtein_distance("abcabc", "", cost=token_edit_distance) == 6
    assert levenshtein_distance("", "abcabc", cost=token_edit_distance) == 6
    assert levenshtein_distance("abcabc", "bcab", cost=token_edit_distance) == 2
    assert levenshtein_distance("abcabc", "abccabc", cost=token_edit_distance) == 1
    assert levenshtein_distance("abcabc", "abacbc", cost=token_edit_distance) == 2
    assert levenshtein_distance("abcabc", "abacbc", cost=token_edit_distance, damerau=True) == 1
    assert levenshtein_distance("abcabc", "abacbca", cost=token_edit_distance, return_oprs=True) == [(SUB, 2, 2), (SUB, 3, 3), (INS, 6, 6)]
    assert levenshtein_distance("abcabc", "abacbca", cost=token_edit_distance, damerau=True, return_oprs=True) == [(TRN, 2, 2), (INS, 6, 6)]

def node_edit_distance(n1, n2, opr):
    if opr == DEL:
        return 1    # insertion
    if opr == INS:
        return 1    # deletion
    return 0 if n1.token == n2.token else 1  # substitution

class _TestNode(object):

    def __init__(self, token):
        self.token = token
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
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(d))"), cost=node_edit_distance) == 0
    assert tree_edit_distance(None, None, cost=node_edit_distance) == 0
    assert tree_edit_distance(tree("a(bc(d))"), None, cost=node_edit_distance) == 4
    assert tree_edit_distance(None, tree("a(bc(d))"), cost=node_edit_distance) == 4
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(cb(d))"), cost=node_edit_distance) == 2
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc)"), cost=node_edit_distance) == 1
    assert tree_edit_distance(tree("a(bc(d))"), tree("a(bc(de))"), cost=node_edit_distance) == 1