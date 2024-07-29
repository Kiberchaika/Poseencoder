import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import gradio as gr
import matplotlib.pyplot as plt
import pickle
import time
from bones_utils import *


def load_3d_keypoints_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        print(f"loaded {len(dataset)} skeletons")
        return dataset

def cluster_skeletons(skeletons, n_clusters, samples_per_cluster=10):
    reshaped_skeletons = skeletons.reshape(skeletons.shape[0], -1)
    
    timestamp = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    cluster_labels = kmeans.fit_predict(reshaped_skeletons)
    print(f"Kmeans elapsed time {time.time() - timestamp} seconds")
    
    # Count skeletons in each cluster and sort clusters by size
    cluster_sizes = [(i, np.sum(cluster_labels == i)) for i in range(n_clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    clustered_skeletons = []
    cluster_data = []
    for cluster_idx, size in cluster_sizes:
        if size < samples_per_cluster:
            continue
        cluster_mask = (cluster_labels == cluster_idx)
        cluster_skeletons = skeletons[cluster_mask]
        clustered_skeletons.append((cluster_skeletons, size))
        
        # Calculate centroid
        centroid = np.mean(cluster_skeletons, axis=0)
        
        # Select random samples
        random_indices = np.random.choice(size, samples_per_cluster, replace=False)
        random_samples = cluster_skeletons[random_indices]
        
        cluster_data.append({
            'cluster_number': cluster_idx,
            'centroid': centroid,
            'random_samples': random_samples
        })
    
    return clustered_skeletons, cluster_data

def rotate_skeleton(skeleton, axis, angle):
    theta = np.radians(angle)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    return np.dot(skeleton, rotation_matrix)

def visualize_skeletons_four_rotations(skeleton, cluster_index, cluster_size):
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f"Cluster {cluster_index} (Size: {cluster_size})", fontsize=16)
    
    skeletons = [
        skeleton,
        rotate_skeleton(skeleton, 'x', 270),
        rotate_skeleton(skeleton, 'y', 45),
        rotate_skeleton(skeleton, 'y', 315)
    ]
    titles = ['Original', 'Upper view 90° (X-axis)', 'View from right 45° (Y-axis)', 'View from left 45° (Y-axis)']
    
    for idx, (ax, rotated_skeleton, title) in enumerate(zip(axs.flatten(), skeletons, titles)):
        # Plot bones
        for start, end in BONES_COCO:
            ax.plot([rotated_skeleton[start, 0], rotated_skeleton[end, 0]],
                    [rotated_skeleton[start, 1], rotated_skeleton[end, 1]], 'bo-', markersize=5, linewidth=2)
        
        # Annotate joints
        for j in range(rotated_skeleton.shape[0]):
            ax.text(rotated_skeleton[j, 0], rotated_skeleton[j, 1], str(j), fontsize=8, ha='right')
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
    
    plt.tight_layout()
    return fig

def gradio_interface(cluster_index, clustered_skeletons):
    cluster_skeletons, cluster_size = clustered_skeletons[cluster_index]
    random_skeleton = cluster_skeletons[np.random.randint(cluster_size)]
    fig = visualize_skeletons_four_rotations(random_skeleton, cluster_index, cluster_size)
    return fig

def save_cluster_data(cluster_data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(cluster_data, f)

if __name__ == "__main__":
    # Load the skeleton data
    skeletons = load_3d_keypoints_dataset("/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed_3d_keypoints/3d_skeletons_dataset.pkl")

    # Cluster the skeletons
    n_clusters = 1000
    samples_per_cluster = 50
    clustered_skeletons, cluster_data = cluster_skeletons(skeletons, n_clusters, samples_per_cluster)

    # print("Saving cluster data")
    # save_cluster_data(cluster_data, "cluster_data_to_umap.pkl")
    # print("Saving cluster data done")

    # Create Gradio interface
    iface = gr.Interface(
        fn=lambda cluster: gradio_interface(cluster, clustered_skeletons),
        inputs=[
            gr.Slider(0, len(clustered_skeletons)-1, step=1, label="Cluster"),
        ],
        outputs="plot",
        title="Random Skeleton Cluster Visualization",
        description="Select a cluster to visualize. Clusters are ordered by size, with the largest cluster having index 0. Each time you submit, a random skeleton from the selected cluster will be shown with its original view and three rotated views.",
    )

    # Generate HTML file
    iface.launch(share=True)