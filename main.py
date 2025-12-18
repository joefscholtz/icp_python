import trimesh
import numpy as np
import sys
import matplotlib.pyplot as plt
import pyglet
from math import sin, cos, atan2
from scipy.spatial import KDTree

from icp import frame_to_frame_icp
from correspondence import correspondence_kdtree
from math_utils import export_trajectories, plot_results, plot_results_3d
from minimization import (
    minimize_point_to_point_svd,
    minimize_point_to_point_ls,
    minimize_point_to_plane_ls,
    minimize_generalized_icp_ls,
)


def main():
    ground_truth = np.load("ground_truth.npy")

    scene = trimesh.Scene()

    N = 30
    X = []
    for i in range(N):
        with open(f"KITTI-Sequence/0000{i:02.0f}/0000{i:02.0f}_points.obj", "r") as f:
            points = trimesh.load(f, file_type="obj", process="false").vertices

            X.append(np.array(points).T)

            pc = trimesh.points.PointCloud(points, colors=[255, 0, 0, 255])

        scene.add_geometry(pc)

    # scene.show(point_size=5)
    # T = frame_to_frame_icp(
    #     X,
    #     correspondence_fn=correspondence_kdtree,
    #     minimization_fn=minimize_point_to_point_svd,
    #     iterations=20,
    # )
    # p_gt, T_gt, p_icp, T_icp = export_trajectories(ground_truth, T)
    # plot_results(p_gt, p_icp)

    T_point_ls = frame_to_frame_icp(
        X,
        correspondence_fn=correspondence_kdtree,
        minimization_fn=minimize_point_to_plane_ls,
        iterations=20,
    )
    p_gt, T_gt, p_icp_point_ls, T_icp_point_ls = export_trajectories(
        ground_truth, T_point_ls
    )
    plot_results(p_gt, p_icp_point_ls)


if __name__ == "__main__":
    main()
