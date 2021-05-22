import numpy as np
import open3d as o3d
import sys

pcd_path = sys.argv[1]
point_cloud = o3d.io.read_point_cloud(pcd_path)
point_cloud = point_cloud.uniform_down_sample(every_k_points=4)
result = point_cloud.cluster_dbscan(eps=0.1, min_points=100)
colors = np.random.rand(30, 3)
point_cloud.colors = o3d.cpu.pybind.utility.Vector3dVector(colors[np.array(result) + 1])
o3d.visualization.draw_geometries([point_cloud])