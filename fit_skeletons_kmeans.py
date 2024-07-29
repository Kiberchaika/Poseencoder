import os
import numpy as np
import argparse
from umap_landmarks_encoder import UMAPLandmarksEncoder
import random
import pickle
import time
from generate_images_grid import generate_embedding_images, generate_embedding_images_kmeans
import bones_utils


def load_motions_data(skeletons_path_3d):
    # Load the pickled cluster data
    with open(skeletons_path_3d, 'rb') as f:
        cluster_data = pickle.load(f)
    centroids = []
    all_skeletons = []
    
    for cluster in cluster_data:
        centroids.append(cluster['centroid'])
        all_skeletons.extend(cluster['random_samples'])
    
    all_skeletons_array = np.array(all_skeletons)
    centroids = np.array(centroids)
    print(f"Loaded {all_skeletons_array.shape[0]} skeletons from {len(cluster_data)} clusters")
    
    return centroids, all_skeletons_array
    

def main(dataset_path):
    # Define keypoint indices for each body part
    upper_body_indices = [0, 5, 6]
    lower_body_indices = [11, 12, 13, 14, 15, 16]
    l_arm_body_indices = [5, 7, 9, 11]
    r_arm_body_indices = [6, 8, 10, 12]

    # Create encoder instances
    upper_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(upper_body_indices), 3), embedding_shape=(2,))
    lower_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(lower_body_indices), 3), embedding_shape=(2,))
    l_arm_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(l_arm_body_indices), 3), embedding_shape=(2,))
    r_arm_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(r_arm_body_indices), 3), embedding_shape=(2,))

    print("Loading data...")
    centroids, skeletons = load_motions_data(dataset_path)
    print("Loading data done")

    print("Adding data to encoders...")
    for skeleton in skeletons:
        upper_encoder.add(skeleton[upper_body_indices])
        lower_encoder.add(skeleton[lower_body_indices])
        l_arm_encoder.add(skeleton[l_arm_body_indices])
        r_arm_encoder.add(skeleton[r_arm_body_indices])
    print("Adding data to encoders done")

    print("Fitting models...")
    upper_encoder.fit()
    lower_encoder.fit()
    l_arm_encoder.fit()
    r_arm_encoder.fit()
    print("Fitting models done")

    # Example skeleton for encoding (using the first centroid)
    example_skeleton = centroids[0]

    print("Encoding example skeleton...")
    ts = time.time()
    for _ in range(1000):
        upper_embedding = upper_encoder.encode(example_skeleton[upper_body_indices])
        lower_embedding = lower_encoder.encode(example_skeleton[lower_body_indices])
        l_arm_embedding = l_arm_encoder.encode(example_skeleton[l_arm_body_indices])
        r_arm_embedding = r_arm_encoder.encode(example_skeleton[r_arm_body_indices])
    print(f"Encoding time: {time.time() - ts} seconds")

    print("Saving models...")
    upper_encoder.save("upper_encoder_model_umap_kmeans.pkl")
    lower_encoder.save("lower_encoder_model_umap_kmeans.pkl")
    l_arm_encoder.save("l_arm_encoder_model_umap_kmeans.pkl")
    r_arm_encoder.save("r_arm_encoder_model_umap_kmeans.pkl")
    print("Saving models done")

    # print("Loading models...")
    # upper_encoder.load("upper_encoder_model_umap_kmeans.pkl")
    # lower_encoder.load("lower_encoder_model_umap_kmeans.pkl")
    # l_arm_encoder.load("l_arm_encoder_model_umap_kmeans.pkl")
    # r_arm_encoder.load("r_arm_encoder_model_umap_kmeans.pkl")
    # print("Loading models done")

    # Generate embeddings for centroid skeletons
    upper_embedding_centroids = [upper_encoder.encode(centroid[upper_body_indices]) for centroid in centroids]
    lower_embeddings_centroids = [lower_encoder.encode(centroid[lower_body_indices]) for centroid in centroids]
    l_arm_embeddings_centroids = [l_arm_encoder.encode(centroid[l_arm_body_indices]) for centroid in centroids]
    r_arm_embeddings_centroids = [r_arm_encoder.encode(centroid[r_arm_body_indices]) for centroid in centroids]

    print("Generating embedding images...")
    os.makedirs("emb", exist_ok=True)
    for i in range(1):
        generate_embedding_images_kmeans(
            centroids,
            upper_embedding_centroids, lower_embeddings_centroids,
            l_arm_embeddings_centroids, r_arm_embeddings_centroids,
            upper_body_indices, lower_body_indices,
            l_arm_body_indices, r_arm_body_indices,
            f"emb/upper_emb/upper_emb_{i}.jpg", f"emb/lower_emb/lower_emb_{i}.jpg",
            f"emb/l_arm_emb/l_arm_emb_{i}.jpg", f"emb/r_arm_emb/r_arm_emb_{i}.jpg",
            grid_size=(32, 32), image_size=(64, 64)
        )
    print("Generating embedding images done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit UMAP encoders on skeleton data")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    args = parser.parse_args()
    main(args.dataset_path)