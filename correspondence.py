import sys
import numpy as np
from scipy.spatial import KDTree


def correspondence_nn(P, Q, max_dist=None, **kwargs):
    correspondences = []
    for i in range(P.shape[1]):
        p = P[:, i]
        dmin = sys.maxsize
        idx = -1
        for j in range(Q.shape[1]):
            q = Q[:, j]
            d = np.linalg.norm(p - q)
            if d < dmin:
                dmin = d
                idx = j
        if max_dist is None or dmin < max_dist:
            correspondences.append((i, idx))
    return correspondences


def correspondence_nn2(P, Q, max_dist=None, **kwargs):
    correspondences = []
    for i in range(P.shape[1]):
        d = np.linalg.norm(Q.T - P[:, i], axis=1)
        j = np.argmin(d)
        if max_dist is None or d[j] < max_dist:
            correspondences.append((i, j))
    return correspondences


def correspondence_kdtree(P, Q, max_dist=None, **kwargs):
    tree = KDTree(Q.T)
    dists, idxs = tree.query(P.T)

    correspondences = []
    for i, (d, j) in enumerate(zip(dists, idxs)):
        if max_dist is None or d < max_dist:
            correspondences.append((i, j))
    return correspondences
