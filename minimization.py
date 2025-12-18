import numpy as np
from math_utils import T_to_Rt, skew, exp_se3


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


def kernel_none(e):
    return 1.0


def kernel_huber(e, delta=1.0):
    r = abs(e)
    if r <= delta:
        return 1.0
    return delta / r


def minimize_point_to_point_ls(
    P, Q, correspondences, iterations=10, kernel=kernel_none, **kwargs
):
    T = np.eye(4)

    for it in range(iterations):
        H = np.zeros((6, 6))
        g = np.zeros((6, 1))
        chi = 0.0

        R = T[:3, :3]
        t = T[:3, 3:4]

        for i, j in correspondences:
            p = P[:, [i]]
            q = Q[:, [j]]

            r = R @ p + t - q

            e = np.linalg.norm(r)
            w = kernel(e)

            J = np.zeros((3, 6))
            J[:, :3] = np.eye(3)
            J[:, 3:] = -R @ skew(p.flatten())

            H += w * (J.T @ J)
            g += w * (J.T @ r)

            chi += (r.T @ r).item()

        dx = np.linalg.solve(H, -g)

        T = exp_se3(dx.flatten()) @ T

    R, t = T_to_Rt(T)

    return R, t, chi
