import numpy as np


def minimize_point_to_point_svd(P, Q, correspondences, **kwargs):
    Ps, Qs = [], []
    for i, j in correspondences:
        Ps.append(P[:, i])
        Qs.append(Q[:, j])

    Ps = np.stack(Ps, axis=1)
    Qs = np.stack(Qs, axis=1)

    mu_P = Ps.mean(axis=1, keepdims=True)
    mu_Q = Qs.mean(axis=1, keepdims=True)

    X = Ps - mu_P
    Y = Qs - mu_Q

    H = X @ Y.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_Q - R @ mu_P

    chi = np.linalg.norm(R @ Ps + t - Qs)

    return R, t, chi
