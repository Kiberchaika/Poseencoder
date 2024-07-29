from torch.utils.data import Dataset
from pose_3d_perspective_projection import load_3d_keypoints_dataset, rotate_skeleton, randomize_limbs, perspective_projection
import numpy as np

# Symmetrical bone pairs
SYMMETRICAL_BONES = [
    (4, 3), (2, 1),  
    (6, 5), (8, 7), (10, 9), 
    (12, 11),  
    (14, 13), (16, 15)
]

class PosesDataset(Dataset):
    def __init__(self, how_many_2d_poses_to_generate, use_additional_augment, split: str):
        self.skeletons = load_3d_keypoints_dataset("/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed_3d_keypoints/3d_skeletons_dataset.pkl")
        self.how_many_2d_poses_to_generate = how_many_2d_poses_to_generate
        self.use_additional_augment = use_additional_augment

        # Split the data into training and testing sets
        split_idx = int(len(self.skeletons) * 0.95)
        self.skeletons = self.skeletons[:split_idx] if split == "train" else self.skeletons[split_idx:]
    
    def __len__(self):
        return len(self.skeletons)
    
    def project_to_2d_and_normalize(self, skeleton):
        x_angle = np.random.uniform(-75, 0)
        y_angle = np.random.uniform(-180, 180)
        z_angle = np.random.uniform(-35, 35)
        
        fov = np.random.uniform(60, 80)

        random_left = False
        random_right = False

        if -130 < y_angle < -60:
            random_left = True
        elif 60 < y_angle < 130:
            random_right = True

        skeleton = randomize_limbs(skeleton, random_left, random_right)
        rotated_skeleton = rotate_skeleton(skeleton, x_angle, y_angle, z_angle)
        projected_skeleton = perspective_projection(rotated_skeleton, fov, 1, 0.1, 100, 5)

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
            random_left = np.random.uniform(0, 1) < 0.25
            random_right = np.random.uniform(0, 1) < 0.25
            skeleton = randomize_limbs(skeleton, random_left, random_right)
 
            # Streach
            skeleton[:, :2] *= [np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)]

            # Flip 
            if np.random.uniform(0, 1) > 0.5:
                skeleton[:, 0] *= -1
                for bone in SYMMETRICAL_BONES:
                    idx1 = bone[0]
                    idx2 = bone[1]
                    skeleton[idx1, :], skeleton[idx2, :] = skeleton[idx2, :].copy(), skeleton[idx1, :].copy()
            
            # Apply noise for 3D
            #skeleton += 0.01 * np.random.uniform(-1, 1, skeleton.shape)

        skeletons_2d_list = []
        for i in range(self.how_many_2d_poses_to_generate):
            normalized_points = self.project_to_2d_and_normalize(skeleton).astype(np.float32)

            #if self.use_additional_augment:
            #    # Apply noise for 2D
            #    normalized_points +=  0.01 * np.random.uniform(-1, 1, normalized_points[:, :2].shape)
 
            skeletons_2d_list.append(normalized_points)

        return { "skeleton_3d" : skeleton, "skeletons_2d" : np.array(skeletons_2d_list) }, 0.0 # dummy data to prevent breaking pytorch lightning 

if __name__ == "__main__":

    dataset = PosesDataset(how_many_2d_poses_to_generate=2, use_additional_augment=False, split='train')
    item = dataset[0]
    print(item[0]['skeleton_3d'].shape)
    print(item[0]['skeletons_2d'].shape)


    