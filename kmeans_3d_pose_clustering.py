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
        
        
# Static variable for COCO bones
BONES_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def cluster_skeletons(skeletons, n_clusters):
    reshaped_skeletons = skeletons.reshape(skeletons.shape[0], -1)
    
    timestamp = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    cluster_labels = kmeans.fit_predict(reshaped_skeletons)
    print(f"Kmeans elapsed time {time.time() - timestamp} seconds")
    
    # Count skeletons in each cluster and sort clusters by size
    cluster_sizes = [(i, np.sum(cluster_labels == i)) for i in range(n_clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    clustered_skeletons = []
    for cluster_idx, size in cluster_sizes:
        cluster_mask = (cluster_labels == cluster_idx)
        cluster_skeletons = skeletons[cluster_mask]
        clustered_skeletons.append((cluster_skeletons, size))
    
    return clustered_skeletons

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

def visualize_skeletons(skeleton, cluster_index, cluster_size):
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f"Cluster {cluster_index} (Size: {cluster_size})", fontsize=16)
    
    skeletons = [
        skeleton,
        rotate_skeleton(skeleton, 'x', 270),
        rotate_skeleton(skeleton, 'y', 45),
        rotate_skeleton(skeleton, 'y', 315)
    ]
    titles = ['Original', 'Upper view 90° (X-axis)', 'View from right 45° (Y-axis)', 'View from right 45° (Y-axis)']
    
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
    fig = visualize_skeletons(random_skeleton, cluster_index, cluster_size)
    return fig

# Generate some random skeleton data for demonstration
# num_skeletons = 1000
# skeletons = np.random.rand(num_skeletons, 17, 3) * 100
skeletons = load_3d_keypoints_dataset("/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed_3d_keypoints/3d_skeletons_dataset.pkl")

# Cluster the skeletons
n_clusters = 1000
clustered_skeletons = cluster_skeletons(skeletons, n_clusters)

# Create Gradio interface
iface = gr.Interface(
    fn=lambda cluster: gradio_interface(cluster, clustered_skeletons),
    inputs=[
        gr.Slider(0, n_clusters-1, step=1, label="Cluster"),
    ],
    outputs="plot",
    title="Random Skeleton Cluster Visualization",
    description="Select a cluster to visualize. Clusters are ordered by size, with the largest cluster having index 0. Each time you submit, a random skeleton from the selected cluster will be shown with its original view and three rotated views.",
)

# Generate HTML file
iface.launch(share=True)