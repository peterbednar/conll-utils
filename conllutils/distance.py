import heapq
import numpy as np

from . import FORM
from . import Sentence, DependencyTree, Instance

DEL = "del"
INS = "ins"
SUB = "sub"
TRN = "trn"

def _normalize(dist, n, m):
    return 2*dist / (n + m + dist) if dist != 0 else 0

def default_token_cost(t1, t2, opr):
    if opr == DEL:
        return 1    # insertion
    if opr == INS:
        return 1    # deletion
    if opr == TRN:
        return 1    # transposition
    return 0 if t1[FORM] == t2[FORM] else 1  # substitution

def levenshtein_distance(s1, s2, cost=default_token_cost, damerau=False, normalize=False, return_oprs=False):
    def _equals(t1, t2):
        return cost(t1, t2, SUB) == 0

    if isinstance(s1, Instance):
        s1 = list(s1.tokens())
    if isinstance(s2, Instance):
        s2 = list(s2.tokens())

    n = len(s1)
    m = len(s2)
    d = np.zeros((n + 1, m + 1), dtype=np.float)

    for i in range(1, n + 1):
        d[i, 0] = d[i-1, 0] + cost(s1[i-1], None, DEL)     # deletion
    for j in range(1, m + 1):
        d[0, j] = d[0, j-1] + cost(None, s2[j-1], INS)     # insertion

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(
                d[i-1, j] + cost(s1[i-1], None, DEL),      # deletion
                d[i, j-1] + cost(None, s2[j-1], INS),      # insertion
                d[i-1, j-1] + cost(s1[i-1], s2[j-1], SUB)  # substitution
            )
            if damerau and i > 1 and j > 1 and _equals(s1[i-1], s2[j-2]) and _equals(s1[i-2], s2[i-1]):
                d[i, j] = min(
                    d[i, j],
                    d[i-2, j-2] + cost(s1[i-2], s2[i-2], TRN)  # transposition
                )
    if not return_oprs:
        if normalize:
            return _normalize(d[n, m], n, m)
        else:
            return d[n, m]

    i = n
    j = m
    oprs = []
    while i > 0 or j > 0:
        neighbours = []
        if damerau and i > 1 and j > 1 and _equals(s1[i-1], s2[j-2]) and _equals(s1[i-2], s2[i-1]):
            neighbours.append((TRN, i-2, j-2))
        if i > 0 and j > 0:
            neighbours.append((SUB, i-1, j-1))
        if j > 0:
            neighbours.append((INS, i, j-1))
        if i > 0:
            neighbours.append((DEL, i-1, j))
        opr = min(neighbours, key=lambda x: d[x[1], x[2]])
        if d[opr[1], opr[2]] != d[i, j]:
            if opr[0] == DEL:
                oprs.append((DEL, opr[1]))
            elif opr[0] == INS:
                oprs.append((INS, opr[2]))
            else:
                oprs.append(opr)
        i = opr[1]
        j = opr[2]
    oprs.reverse()

    if normalize:
        return _normalize(d[n, m], n, m), oprs
    else:
        return d[n, m], oprs

class _AnnotatedNode(object):

    def __init__(self, node):
        self.node = node
        self.index = -1
        self.leftmost = None
        self.children = []

    def collect(self):
        return self._collect([], [])

    def _collect(self, nodes, l):
        for child in self.children:
            nodes, l = child._collect(nodes, l)
        nodes.append(self.node)
        l.append(self.leftmost.index)
        return nodes, l

    @staticmethod
    def build(node, index=0):
        anode = _AnnotatedNode(node)
        for child in node.children:
            achild = _AnnotatedNode.build(child, index)
            index = achild.index + 1
            anode.children.append(achild)
        anode.index = index
        if anode.children:
            anode.leftmost = anode.children[0].leftmost
        else:
            anode.leftmost = anode
        return anode

def _annotate(root):
    if root is None:
        return [], [], []        
    nodes, l = _AnnotatedNode.build(root).collect()
    keyroots = []
    n = len(l)
    for i in range(n):
        is_root = True
        for j in range(i + 1, n):
            if l[i] == l[j]:
                is_root = False
                break
        if is_root:
            keyroots.append(i)
    return nodes, l, keyroots

def _treedist(i, j, l1, l2, nodes1, nodes2, TD, TD_oprs, cost, return_oprs):

    def _merge(x1, y1, x2, y2, l):
        d_oprs[x1, y1] = d_oprs[x2, y2] + l

    n = i - l1[i] + 2
    m = j - l2[j] + 2
    d = np.zeros((n, m), dtype=np.float)
    if return_oprs:
        d_oprs = np.empty((n, m), dtype=np.object)
        d_oprs.fill([])
    i_off = l1[i] - 1
    j_off = l2[j] - 1

    for x in range(1, n):
        d[x, 0] = d[x-1, 0] + cost(nodes1[x+i_off], None, DEL)      # delete
        if return_oprs:
            _merge(x, 0, x-1, 0, [(DEL, x+i_off)])

    for y in range(1, m):
        d[0, y] = d[0, y-1] + cost(None, nodes2[y+j_off], INS)      # insert
        if return_oprs:
            _merge(0, y, 0, y-1, [(INS, y+j_off)])

    for x in range(1, n):
        for y in range(1, m):
            xi = x + i_off
            yj = y + j_off
            if l1[i] == l1[xi] and l2[j] == l2[yj]:
                costs = (
                    d[x-1, y] + cost(nodes1[xi], None, DEL),        # delete
                    d[x, y-1] + cost(None, nodes2[yj], INS),        # insert
                    d[x-1, y-1] + cost(nodes1[xi], nodes2[yj], SUB) # substitute
                )
                min_cost = min(costs)
                d[x, y] = min_cost
                TD[xi, yj] = d[x, y]
                if return_oprs:
                    if min_cost == costs[0]:
                        _merge(x, y, x-1, y, [(DEL, xi)])
                    elif min_cost == costs[1]:
                        _merge(x, y, x, y-1, [(INS, yj)])
                    else:
                        opr = [(SUB, xi, yj)] if d[x, y] != d[x-1, y-1] else []
                        _merge(x, y, x-1, y-1, opr)
                    TD_oprs[xi, yj] = d_oprs[x, y]
            else:
                x_tmp = l1[xi]-1-i_off
                y_tmp = l2[yj]-1-j_off
                costs = (
                    d[x-1, y] + cost(nodes1[xi], None, DEL),
                    d[x, y-1] + cost(None, nodes2[yj], INS),
                    d[x_tmp, y_tmp] + TD[xi, yj]
                )
                min_cost = min(costs)
                d[x, y] = min_cost
                if return_oprs:
                    if min_cost == costs[0]:
                        _merge(x, y, x-1, y, [(DEL, xi)])
                    elif min_cost == costs[1]:
                        _merge(x, y, x, y-1, [(INS, yj)])
                    else:
                        _merge(x, y, x_tmp, y_tmp, TD_oprs[xi, yj])

def default_node_cost(n1, n2, opr):
    t1 = None if n1 is None else n1.token
    t2 = None if n2 is None else n2.token
    return default_token_cost(t1, t2, opr)

def tree_edit_distance(t1, t2, cost=default_node_cost, normalize=False, return_oprs=False):

    def _get_root(t):
        if isinstance(t, DependencyTree):
            return t.root
        if isinstance(t, Sentence):
            return t.as_tree().root
        return t

    nodes1, l1, keyroots1 = _annotate(_get_root(t1))
    nodes2, l2, keyroots2 = _annotate(_get_root(t2))

    n = len(nodes1)
    m = len(nodes2)

    if n == 0 and m == 0:
        dist = 0
        if return_oprs:
            oprs = []
    elif n != 0 and m == 0:
        dist = sum(cost(node, None, DEL) for node in nodes1)
        if return_oprs:
            oprs = [(DEL, i) for i in range(n)]
    elif n == 0 and m != 0:
        dist = sum(cost(None, node, INS) for node in nodes2)
        if return_oprs:
            oprs = [(INS, j) for j in range(m)]
    else:
        TD = np.zeros((n, m), dtype=np.float)
        if return_oprs:
            TD_oprs = np.empty((n, m), dtype=np.object)
            TD_oprs.fill([])
        else:
            TD_oprs = None

        for i in keyroots1:
            for j in keyroots2:
                _treedist(i, j, l1, l2, nodes1, nodes2, TD, TD_oprs, cost, return_oprs)

        dist = TD[n-1, m-1]
        if return_oprs:
            oprs = TD_oprs[n-1, m-1]

    if normalize:
        dist = _normalize(dist, n, m)

    if return_oprs:
        return dist, oprs
    else:
        return dist

def default_value_cost(key, v1, v2, opr):
    if opr == DEL:
        return 1
    if opr == INS:
        return 1
    return 1 if v1 != v2 else 0

from . import _parse_feats

def dict_edit_distance(d1, d2, cost=default_value_cost, normalize=False, return_oprs=False):

    if isinstance(d1, str):
        d1 = _parse_feats(d1)
    if isinstance(d2, str):
        d2 = _parse_feats(d2)

    all_keys = set(d1.keys()).union(d2.keys())
    dist = 0
    oprs = set()

    for key in all_keys:
        if key not in d2:
            c = cost(key, d1[key], None, DEL)   # delete
            opr = (DEL, key)
        elif key not in d1:
            c = cost(key, None, d2[key], INS)   # insert
            opr = (INS, key)
        else:
            c = cost(key, d1[key], d2[key], SUB)    # substitute
            opr = (SUB, key)
        if c != 0:
            dist += c
            if return_oprs:
                oprs.add(opr)

    if normalize:
        dist = dist / len(all_keys) if dist != 0 else 0
    if return_oprs:
        return dist, oprs
    else:
        return dist

def k_nearest_neighbors(itm1, itms2, k=1, distance=levenshtein_distance, return_distance=True):
    knn = heapq.nsmallest(k, [(idx2, distance(itm1, itm2)) for idx2, itm2 in enumerate(itms2)], key=lambda x: x[1])
    if return_distance:
        return knn
    else:
        return [itm2[0] for itm2 in knn]
