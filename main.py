import numpy as np
import open3d as o3d
import sys
import os
import json

MARGIN = -0.32
    
def load_and_process_clouds(clouds_path):
    for cloud_name in os.listdir(clouds_path):
        cloud_path = os.path.join(clouds_path, cloud_name)
        cloud = o3d.io.read_point_cloud(cloud_path)
        cloud_result = process_cloud(cloud)
        write_output(cloud_result, clouds_path, cloud_name)

def get_clusters(points, clustering):
    clusters = {cluster_idx : [] for cluster_idx in np.unique(clustering)}
    for i in range(len(points)):
        clusters[clustering[i]].append(points[i])
    for cluster_idx, cluster in clusters.items():
        clusters[cluster_idx] = np.array(cluster)
    return clusters
        
def is_cluster_violating(cluster, margin=MARGIN):
    return 0.05 < np.mean(cluster[:, 1] < MARGIN) < 0.95

def process_cloud(cloud):
    downsampled_cloud = cloud.uniform_down_sample(every_k_points=4)
    clustering = downsampled_cloud.cluster_dbscan(eps=0.1, min_points=100)
    clusters = get_clusters(downsampled_cloud.points, clustering)
    
    cloud_result = {}
    violation = np.any(np.array([is_cluster_violating(cluster) for cluster in clusters.values()]))
    cloud_result['num_clusters'] = len(clusters)
    cloud_result['is_safe'] = 'no' if violation else 'yes'
    return cloud_result

def write_output(cloud_result, clouds_path, cloud_name):
    output_path = os.path.join(clouds_path, 'output/')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    outfile_path = os.path.splitext(os.path.join(output_path, cloud_name))[0] + '.json'
    with open(outfile_path, 'w') as outfile:
        json.dump(cloud_result, outfile)
        
load_and_process_clouds(clouds_path=sys.argv[1])
    
    
    


