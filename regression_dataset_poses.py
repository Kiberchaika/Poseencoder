from torch.utils.data import Dataset
from pose_3d_perspective_projection import load_3d_keypoints_dataset, rotate_skeleton, randomize_limbs, perspective_projection
from umap_landmarks_encoder import UMAPLandmarksEncoder
import numpy as np

# Symmetrical bone pairs
SYMMETRICAL_BONES = [
    (4, 3), (2, 1),  
    (6, 5), (8, 7), (10, 9), 
    (12, 11),  
    (14, 13), (16, 15)
]

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
        # V1
        # x_angle = np.random.uniform(-75, 0)
        # y_angle = np.random.uniform(-180, 180)
        # z_angle = np.random.uniform(-35, 35)
        
        x_angle = np.random.uniform(-10, 10)
        y_angle = np.random.uniform(-40, 40)
        z_angle = np.random.uniform(-10, 10)

        fov = np.random.uniform(60, 80)

        random_left = False
        random_right = False

        if -130 < y_angle < -60:
            random_left = True
        elif 60 < y_angle < 130:
            random_right = True

        # skeleton = randomize_limbs(skeleton, random_left, random_right)
        rotated_skeleton = rotate_skeleton(skeleton, x_angle, y_angle, z_angle)
        projected_skeleton = perspective_projection(rotated_skeleton, fov, 1, 0.1, 100, 15)

        # normalisation for 2d -1:1
        distances = np.sqrt(np.sum(projected_skeleton**2, axis=1))
        max_distance = np.max(distances)
        normalized_points = projected_skeleton / max_distance

        return normalized_points

    def __getitem__(self, idx):
        skeleton = self.skeletons[idx].copy().astype(np.float32)
        
        if self.use_additional_augment:
            # todo: add some augmentations 
            # ? менять обе руки или обе ноги для 3д
            # ? вращать немного руки, ноги, голову для 3д

            # Random swap
            #random_left = np.random.uniform(0, 1) < 0.25
            #random_right = np.random.uniform(0, 1) < 0.25
            #skeleton = randomize_limbs(skeleton, random_left, random_right)
 
            # Streach
            skeleton[:, :2] *= [np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)]

            # Flip
            ''' 
            if np.random.uniform(0, 1) > 0.5:
                skeleton[:, 0] *= -1
                for bone in SYMMETRICAL_BONES:
                    idx1 = bone[0]
                    idx2 = bone[1]
                    skeleton[idx1, :], skeleton[idx2, :] = skeleton[idx2, :].copy(), skeleton[idx1, :].copy()
            '''
            # Apply noise for 3D
            #skeleton += 0.01 * np.random.uniform(-1, 1, skeleton.shape)

        normalized_points = self.project_to_2d_and_normalize(skeleton).astype(np.float32)

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

        return {
            "embeddings": embeddings,
            "skeleton_2d": normalized_points
        }, 0.0  # dummy data to prevent breaking pytorch lightning

if __name__ == "__main__":
    dataset = PosesDataset(use_additional_augment=False, split='train')
    item = dataset[0]
    print("2D Skeleton shape:", item[0]['skeleton_2d'].shape)
    print("Embeddings:")
    for key, value in item[0]['embeddings'].items():
        print(f"  {key} shape:", value.shape)


    