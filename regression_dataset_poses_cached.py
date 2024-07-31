import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
from pose_3d_perspective_projection import load_3d_keypoints_dataset, rotate_skeleton, randomize_limbs, perspective_projection
from umap_landmarks_encoder import UMAPLandmarksEncoder

class PosesDataset(Dataset):
    def __init__(self, use_additional_augment, split: str):
        self.skeletons = load_3d_keypoints_dataset("/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed_3d_keypoints/3d_skeletons_dataset.pkl")
        self.use_additional_augment = use_additional_augment

        # Split the data into training and testing sets
        split_idx = int(len(self.skeletons) * 0.95)
        self.skeletons = self.skeletons[:split_idx] if split == "train" else self.skeletons[split_idx:]

        self.upper_body_indices = [4, 5, 6]
        self.lower_body_indices = [11, 12, 13, 14, 15, 16]
        self.l_arm_body_indices = [5, 7, 9, 11]
        self.r_arm_body_indices = [6, 8, 10, 12]

        self.upper_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(self.upper_body_indices), 3), embedding_shape=(2,))
        self.lower_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(self.lower_body_indices), 3), embedding_shape=(2,))
        self.l_arm_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(self.l_arm_body_indices), 3), embedding_shape=(2,))
        self.r_arm_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(self.r_arm_body_indices), 3), embedding_shape=(2,))

        self.load_umaps()
        
        # Initialize the cache
        self.cache_file = f"regression_dataset_caches_{split}.pkl"
        self.embeddings_cache = self.load_cache()
        self.cache_modified = False

    def load_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Loading cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("Cache file not found. Creating a new cache.")
            return [None] * len(self.skeletons)

    def save_cache(self):
        if self.cache_modified:
            print(f"Saving cache to {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            self.cache_modified = False

    def load_umaps(self):
        print("Loading models...")
        self.lower_encoder.load("lower_encoder_model_umap_kmeans.pkl")
        self.upper_encoder.load("upper_encoder_model_umap_kmeans.pkl")
        self.l_arm_encoder.load("l_arm_encoder_model_umap_kmeans.pkl")
        self.r_arm_encoder.load("r_arm_encoder_model_umap_kmeans.pkl")
        print("Loading models done")

    def __len__(self):
        return len(self.skeletons)
    
    def project_to_2d_and_normalize(self, skeleton):
        x_angle = np.random.uniform(-10, 10)
        y_angle = np.random.uniform(-40, 40)
        z_angle = np.random.uniform(-10, 10)

        fov = np.random.uniform(60, 80)

        rotated_skeleton = rotate_skeleton(skeleton, x_angle, y_angle, z_angle)
        projected_skeleton = perspective_projection(rotated_skeleton, fov, 1, 0.1, 100, 15)

        distances = np.sqrt(np.sum(projected_skeleton**2, axis=1))
        max_distance = np.max(distances)
        normalized_points = projected_skeleton / max_distance

        return normalized_points

    def __getitem__(self, idx):
        skeleton = self.skeletons[idx].copy().astype(np.float32)
        
        if self.use_additional_augment:
            skeleton[:, :2] *= [np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)]

        normalized_points = self.project_to_2d_and_normalize(skeleton).astype(np.float32)

        # Check if embeddings are in cache
        if self.embeddings_cache[idx] is None:
            upper_embedding = self.upper_encoder.encode(skeleton[self.upper_body_indices])
            lower_embedding = self.lower_encoder.encode(skeleton[self.lower_body_indices])
            l_arm_embedding = self.l_arm_encoder.encode(skeleton[self.l_arm_body_indices])
            r_arm_embedding = self.r_arm_encoder.encode(skeleton[self.r_arm_body_indices])

            embeddings = {
                "upper": upper_embedding,
                "lower": lower_embedding,
                "l_arm": l_arm_embedding,
                "r_arm": r_arm_embedding
            }
            
            # Save to cache
            self.embeddings_cache[idx] = embeddings

            self.cache_modified = True
        else:
            embeddings = self.embeddings_cache[idx]

        return {
            "embeddings": embeddings,
            "skeleton_2d": normalized_points
        }, 0.0  # dummy data to prevent breaking pytorch lightning

    def __del__(self):
        # Save cache when the dataset object is destroyed
        self.save_cache()

    def generate_cache(self):
        print("Generating cache for all data points...")
        for idx in tqdm(range(len(self))):
            _ = self[idx]  # This will compute and cache the embeddings
        self.save_cache()
        print("Cache generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PosesDataset with cache generation option")
    parser.add_argument('--generate_cache', action='store_true', help='Generate cache for all data points')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
    args = parser.parse_args()

    dataset = PosesDataset(use_additional_augment=False, split=args.split)

    if args.generate_cache:
        dataset.generate_cache()
    else:
        # Normal dataset usage
        item = dataset[0]
        print("2D Skeleton shape:", item[0]['skeleton_2d'].shape)
        print("Embeddings:")
        for key, value in item[0]['embeddings'].items():
            print(f"  {key} shape:", value.shape)

    # Always save the cache at the end
    dataset.save_cache()