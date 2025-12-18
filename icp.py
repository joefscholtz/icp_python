import numpy as np
from math_utils import T_to_Rt, Rt_to_T, transform_points


def icp(P, Q, correspondence_fn, minimization_fn, iterations=20, T0=None, **options):
    if T0 is None:
        R = np.eye(3)
        t = np.zeros((3, 1))
    else:
        R, t = T_to_Rt(T0)
    P_curr = transform_points(R, t, P)
    P_vals = [P_curr.T]
    T_vals = []
    chi_vals = []
    for i in range(iterations):
        correspodences = correspondence_fn(P_curr, Q, **options)
        R, t, chi = minimization_fn(P_curr, Q, correspodences, **options)
        P_curr = transform_points(R, t, P)
        P_vals.append(P_curr.T)
        T = Rt_to_T(R, t)
        T_vals.append(T)
        chi_vals.append(chi)

    return P_vals, T_vals, chi_vals


def frame_to_frame_icp(X, correspondence_fn, minimization_fn, iterations=20, **options):
    T_k = Rt_to_T(np.eye(3), np.zeros((3, 1)))
    T_list = [T_k.copy()]

    for k in range(1, len(X)):
        print(f"Computing transform between pcl {k-1} and {k}")

        P = X[k]
        Q = X[k - 1]

        _, T_vals, _ = icp(
            P,
            Q,
            correspondence_fn=correspondence_fn,
            minimization_fn=minimization_fn,
            iterations=iterations,
            T0=None,
            **options,
        )

        Delta_T = T_vals[-1]
        T_k = T_k @ Delta_T

        T_list.append(T_k.copy())

    return np.stack(T_list, axis=0)  # (N, 4, 4)
