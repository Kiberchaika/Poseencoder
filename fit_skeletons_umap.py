import os
import numpy as np
import argparse
from umap_landmarks_encoder import UMAPLandmarksEncoder
import random
import time
from generate_images_grid import generate_embedding_images
import bones_utils

def load_motions_data(collection_path):
    skeletons = []
    for collection_folder_name in sorted(os.listdir(collection_path)):
        collection_folder_path = os.path.join(collection_path, collection_folder_name)
        if os.path.isdir(collection_folder_path):
            for motion_folder in sorted(os.listdir(collection_folder_path)):
                motion_folder_path = os.path.join(collection_folder_path, motion_folder)
                if os.path.isdir(motion_folder_path):
                    keypoints_path = os.path.join(motion_folder_path, "all_keypoints.npy")
                    if os.path.exists(keypoints_path):
                        motion_list = np.load(keypoints_path)
                        for idx, keypoints in enumerate(motion_list):
                            if idx % 50 == 0 and not np.array_equal(keypoints, np.zeros((17, 2))):
                                skeletons.append(keypoints)
    return np.array(skeletons)

def main(dataset_path):
    # Define keypoint indices for each body part
    upper_body_indices = [4, 5, 6]
    lower_body_indices = [11, 12, 13, 14, 15, 16]
    l_arm_body_indices = [5, 7, 9, 11]
    r_arm_body_indices = [6, 8, 10, 12]

    # Create encoder instances
    upper_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(upper_body_indices), 2), embedding_shape=(2,))
    lower_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(lower_body_indices), 2), embedding_shape=(2,))
    l_arm_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(l_arm_body_indices), 2), embedding_shape=(2,))
    r_arm_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(r_arm_body_indices), 2), embedding_shape=(2,))

    print("Loading data...")
    skeletons = load_motions_data(dataset_path)
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

    # Example skeleton for encoding
    example_skeleton = skeletons[0]

    print("Encoding example skeleton...")
    ts = time.time()
    for _ in range(1000):
        upper_embedding = upper_encoder.encode(example_skeleton[upper_body_indices])
        lower_embedding = lower_encoder.encode(example_skeleton[lower_body_indices])
        l_arm_embedding = l_arm_encoder.encode(example_skeleton[l_arm_body_indices])
        r_arm_embedding = r_arm_encoder.encode(example_skeleton[r_arm_body_indices])
    print(f"Encoding time: {time.time() - ts} seconds")

    print("Saving models...")
    upper_encoder.save("upper_encoder_model_umap.pkl")
    lower_encoder.save("lower_encoder_model_umap.pkl")
    l_arm_encoder.save("l_arm_encoder_model_umap.pkl")
    r_arm_encoder.save("r_arm_encoder_model_umap.pkl")
    print("Saving models done")

    # print("Loading models...")
    # upper_encoder.load("upper_encoder_model_umap.pkl")
    # lower_encoder.load("lower_encoder_model_umap.pkl")
    # l_arm_encoder.load("l_arm_encoder_model_umap.pkl")
    # r_arm_encoder.load("r_arm_encoder_model_umap.pkl")
    # print("Loading models done")

    print("Generating embedding images...")
    os.makedirs("emb", exist_ok=True)
    for i in range(5):
        generate_embedding_images(
            skeletons,
            upper_encoder.embeddings, lower_encoder.embeddings,
            l_arm_encoder.embeddings, r_arm_encoder.embeddings,
            upper_body_indices, lower_body_indices,
            l_arm_body_indices, r_arm_body_indices,
            f"emb/upper_emb/upper_emb_{i}.jpg", f"emb/lower_emb/lower_emb_{i}.jpg",
            f"emb/l_arm_emb/l_arm_emb_{i}.jpg", f"emb/r_arm_emb/r_arm_emb_{i}.jpg",
            grid_size=(16, 16), image_size=(64, 64),
            upper_2d_points=upper_embedding,
            lower_2d_points=lower_embedding,
            l_arm_2d_points=l_arm_embedding,
            r_arm_2d_points=r_arm_embedding
        )
    print("Generating embedding images done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit UMAP encoders on skeleton data")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    args = parser.parse_args()
    main(args.dataset_path)