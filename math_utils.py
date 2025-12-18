import numpy as np
import matplotlib.pyplot as plt


def T_to_Rt(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    return R, t


def Rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def transform_points(R, t, P):
    return R @ P + t


def extract_positions(T):
    return T[:, :3, 3]


def estimate_heading(T, M=5):
    assert M >= 2
    assert T.shape[0] >= M

    # Extract positions
    p0 = T[0, :3, 3]
    pM = T[1:M, :3, 3]

    # Mean direction of motion
    direction = np.mean(pM - p0, axis=0)

    # Project to ground plane (XY)
    direction[2] = 0.0

    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("Degenerate heading: motion too small")

    direction /= norm

    yaw = np.arctan2(direction[1], direction[0])
    return yaw, direction


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def translate_to_origin(T_list):
    T0_inv = np.eye(4)
    T0_inv[:3, 3] = -T_list[0][:3, 3]
    return np.array([T0_inv @ T for T in T_list])


def export_trajectories(gt, T):
    T_gt = translate_to_origin(gt)
    p_gt = extract_positions(T_gt)

    yaw_gt, _ = estimate_heading(gt, M=5)
    yaw_icp, _ = estimate_heading(T, M=5)

    delta_yaw = yaw_gt - yaw_icp
    R_align = Rz(delta_yaw)

    T_icp_oriented = []

    T0 = T[0]

    for Tk in T:
        T_new = np.eye(4)

        # rotate orientation
        T_new[:3, :3] = R_align @ Tk[:3, :3]

        # rotate position around first pose
        T_new[:3, 3] = R_align @ (Tk[:3, 3] - T0[:3, 3]) + T0[:3, 3]

        T_icp_oriented.append(T_new)

    T_icp_oriented = np.stack(T_icp_oriented, axis=0)

    T_icp = translate_to_origin(T_icp_oriented)
    p_icp = extract_positions(T_icp)

    return p_gt, T_gt, p_icp, T_icp


def align_trajectories(P, Q):
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)

    X = P - mu_P
    Y = Q - mu_Q

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_Q - R @ mu_P
    return R, t


def compute_ate(P_est, P_gt):
    ate = np.linalg.norm(P_est - P_gt, axis=1)
    ate_rmse = np.sqrt(np.mean(ate**2))

    return ate_rmse, ate


def plot_results(p_gt, p_icp):
    ate_rmse, ate = compute_ate(p_gt, p_icp)

    plt.plot(p_gt[:, 0], p_gt[:, 1], "g-", label="GT")
    plt.plot(p_icp[:, 0], p_icp[:, 1], "r--", label="ICP")
    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 4))

    plt.plot(ate, "r-", label="ATE")
    # plt.plot(cumulative_ate,label="Cumulative ATE")
    plt.xlabel("Frame index")
    plt.ylabel("ATE [m]")
    plt.title("Absolute Trajectory Error per frame")
    plt.grid()
    plt.show()


def plot_results(p_gt, p_icp):
    ate_rmse, ate = compute_ate(p_gt, p_icp)

    plt.plot(p_gt[:, 0], p_gt[:, 1], "g-", label="GT")
    plt.plot(p_icp[:, 0], p_icp[:, 1], "r--", label="ICP")
    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 4))

    plt.plot(ate, "r-", label="ATE")
    # plt.plot(cumulative_ate,label="Cumulative ATE")
    plt.xlabel("Frame index")
    plt.ylabel("ATE [m]")
    plt.title("Absolute Trajectory Error per frame")
    plt.grid()
    plt.show()


def plot_results_3d(p_gt, p_icp):
    ate_rmse, ate = compute_ate(p_icp, p_gt)
    print(f"ATE RMSE: {ate_rmse:.4f} m")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2], "g-", label="Ground Truth")
    ax.plot(p_icp[:, 0], p_icp[:, 1], p_icp[:, 2], "r--", label="ICP")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Trajectory comparison (3D)")
    ax.legend()

    all_pts = np.vstack((p_gt, p_icp))
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    mid = all_pts.mean(axis=0)

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(ate, "r-", label="ATE")
    plt.xlabel("Frame index")
    plt.ylabel("ATE [m]")
    plt.title("Absolute Trajectory Error per frame")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)

    k = w / theta
    K = skew(k)

    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def exp_se3(xi):
    rho = xi[:3].reshape(3, 1)
    w = xi[3:]

    R = exp_so3(w)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rho.flatten()
    return T


def compute_normals(points, k=20):
    P = points.T  # (N,3)
    N = P.shape[0]

    normals = np.zeros((N, 3))

    for i in range(N):
        d = np.linalg.norm(P - P[i], axis=1)
        idx = np.argsort(d)[1 : k + 1]
        neigh = P[idx].T  # (3,k)

        mu = neigh.mean(axis=1, keepdims=True)
        C = (neigh - mu) @ (neigh - mu).T

        _, v = np.linalg.eigh(C)
        n = v[:, 0]  # smallest eigenvalue
        normals[i] = n / np.linalg.norm(n)

    return normals  # (N,3)


def compute_covariances(points, tree, k=20, eps=1e-6):
    P = points.T  # (N,3)
    N = P.shape[0]

    covariances = []

    for i in range(N):
        # k nearest neighbors (excluding the point itself)
        dists, idx = tree.query(P[i], k=k + 1)
        neigh = P[idx[1:]]  # (k,3)

        mu = neigh.mean(axis=0, keepdims=True)
        X = neigh - mu

        C = (X.T @ X) / k

        # Regularization (prevents singular matrices)
        C += eps * np.eye(3)

        covariances.append(C)

    return covariances
