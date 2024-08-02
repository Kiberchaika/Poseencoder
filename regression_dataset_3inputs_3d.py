import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
from pose_3d_perspective_projection import load_3d_keypoints_dataset, rotate_skeleton, randomize_limbs, perspective_projection
import torch

class PosesDataset3cam(Dataset):
    def __init__(self, use_additional_augment, split: str):
        self.skeletons = load_3d_keypoints_dataset("3d_skeletons_dataset_bedlam_v1.pkl")
        self.use_additional_augment = use_additional_augment

        # Split the data into training and testing sets
        split_idx = int(len(self.skeletons) * 0.85)
        self.skeletons = self.skeletons[:split_idx] if split == "train" else self.skeletons[split_idx:]

        self.upper_body_indices = [4, 5, 6]
        self.lower_body_indices = [11, 12, 13, 14, 15, 16]
        self.l_arm_body_indices = [5, 7, 9, 11] # 11 is waist
        self.r_arm_body_indices = [6, 8, 10, 12] # 12 is waist

        print("dataset initialized")

        # # Initialize the cache
        # self.cache_file = f"regression_pca_dataset_caches_{split}.pkl"
        # self.embeddings_cache = [None] * len(self.skeletons)
        # # self.embeddings_cache = self.load_cache()
        # self.cache_modified = False

    # def load_cache(self):
    #     if os.path.exists(self.cache_file):
    #         print(f"Loading cache from {self.cache_file}")
    #         with open(self.cache_file, 'rb') as f:
    #             return pickle.load(f)
    #     else:
    #         print("Cache file not found. Creating a new cache.")
    #         return [None] * len(self.skeletons)

    # def save_cache(self):
    #     if self.cache_modified:
    #         print(f"Saving cache to {self.cache_file}")
    #         with open(self.cache_file, 'wb') as f:
    #             pickle.dump(self.embeddings_cache, f)
    #         self.cache_modified = False


    def __len__(self):
        return len(self.skeletons)
    
    def project_to_2d_augment_and_normalize(self, skeleton):
        x_angle = np.random.uniform(-40, -30)
        y_angle = np.random.uniform(-120, 120)
        z_angle = np.random.uniform(-20, 20)

        fov = np.random.uniform(60, 60)

        rotated_skeleton = rotate_skeleton(skeleton, x_angle, y_angle, z_angle)
        projected_skeleton = perspective_projection(rotated_skeleton, fov, 1, 0.1, 100, 25)

        distances = np.sqrt(np.sum(projected_skeleton**2, axis=1))
        max_distance = np.max(distances)
        normalized_points = projected_skeleton / max_distance

        # Add noise
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, normalized_points.shape)
    
        # With 5% chance, apply higher noise to two random landmarks
        if np.random.random() < 1.0:
            high_noise_indices = np.random.choice(normalized_points.shape[0], 2, replace=False)
            high_noise = np.random.normal(0, noise_scale * 3, (2, 2))
            noise[high_noise_indices] = high_noise

        normalized_points_with_noise = normalized_points + noise

        return normalized_points_with_noise
    
    def return_regularisation_batch(self, batch):
        batch_size = batch['skeleton_3d'].size(0)
        reg_batches = {}
        
        for part, indices in [('upper', self.upper_body_indices), 
                              ('lower', self.lower_body_indices), 
                              ('l_arm', self.l_arm_body_indices), 
                              ('r_arm', self.r_arm_body_indices)]:
            
            # Pick a random sample for the current body part
            random_idx = torch.randint(0, batch_size, (1,)).item()
            
            # Create a copy of the input batch structure
            reg_batch = {
                'embeddings': {k: v.clone() for k, v in batch['embeddings'].items()},
                'skeleton_2d': [],
                'skeleton_3d': []
            }

            # Fix the embedding for the current part across all samples
            fixed_embedding = batch['embeddings'][part][random_idx].clone().detach()
            reg_batch['embeddings'][part] = fixed_embedding.repeat(batch_size, 1)

            fixed_part_3d = batch['skeleton_3d'][random_idx, indices].clone().detach()

            # Randomize all parts except the current one and set the fixed part for all samples
            for i in range(batch_size):
                new_skeleton = batch['skeleton_3d'][i].clone()
                for other_part, other_indices in [('upper', self.upper_body_indices), 
                                                  ('lower', self.lower_body_indices), 
                                                  ('l_arm', self.l_arm_body_indices), 
                                                  ('r_arm', self.r_arm_body_indices)]:
                    if other_part != part:
                        random_skeleton_idx = torch.randint(0, len(self.skeletons), (1,)).item()
                        random_skeleton = torch.tensor(self.skeletons[random_skeleton_idx], dtype=torch.float32)
                        new_skeleton[other_indices] = random_skeleton[other_indices]
                    else:
                        new_skeleton[indices] = fixed_part_3d

                reg_batch['skeleton_3d'].append(new_skeleton)
                
                # Project 3D skeleton to 2D and normalize
                projected_2d = self.project_to_2d_and_normalize(new_skeleton.cpu().numpy())
                reg_batch['skeleton_2d'].append(torch.tensor(projected_2d, dtype=torch.float32))

            # Convert lists to tensors
            reg_batch['skeleton_3d'] = torch.stack(reg_batch['skeleton_3d'])
            reg_batch['skeleton_2d'] = torch.stack(reg_batch['skeleton_2d'])

            reg_batches[part] = reg_batch

        return reg_batches

    def __getitem__(self, idx):
        skeleton = self.skeletons[idx].copy().astype(np.float32)
        
        if self.use_additional_augment:
            skeleton[:, :2] *= [np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)]

        normalized_2d_points = self.project_to_2d_augment_and_normalize(skeleton).astype(np.float32)
        normalized_2d_points_2 = self.project_to_2d_augment_and_normalize(skeleton).astype(np.float32)
        normalized_2d_points_3 = self.project_to_2d_augment_and_normalize(skeleton).astype(np.float32)

        return {
            "skeleton_2d": normalized_2d_points,
            "skeleton_2d_2": normalized_2d_points_2,
            "skeleton_2d_3": normalized_2d_points_3,
            "skeleton_3d": skeleton
        }, 0.0  # dummy data to prevent breaking pytorch lightning

    # def __del__(self):
    #     # Save cache when the dataset object is destroyed
    #     self.save_cache()

    # def generate_cache(self):
    #     print("Generating cache for all data points...")
    #     for idx in tqdm(range(len(self))):
    #         _ = self[idx]  # This will compute and cache the embeddings
    #     self.save_cache()
    #     print("Cache generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PosesDataset with cache generation option")
    parser.add_argument('--generate_cache', action='store_true', help='Generate cache for all data points')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
    args = parser.parse_args()

    dataset = PosesDataset3cam(use_additional_augment=False, split=args.split)

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