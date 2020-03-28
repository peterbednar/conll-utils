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
    return 0 if n1 == n2 else 1  # substitution

def test_tree_edit_distance():
    pass