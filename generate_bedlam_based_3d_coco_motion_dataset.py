import numpy as np
import torch
import smplx
import trimesh
import pyrender
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyvirtualdisplay import Display

import torch
import numpy as np
from PIL import Image
import cv2
import os
import pickle

from detectors.yolo import YOLODetector
from bones_utils import BONES_COCO

from pose_3d_perspective_projection import rotate_project_draw_two_3d_skeleton

HUMAN_MODEL_NAMES = ["male", "female", "neutral"]

def load_motion_sequence(path):
    data = np.load(path)
    return {
        'betas': torch.tensor(data['betas'], dtype=torch.float32),
        'poses': torch.tensor(data['poses'], dtype=torch.float32),
        'trans': torch.tensor(data['trans'], dtype=torch.float32),
        'mocap_frame_rate': data['mocap_frame_rate']
    }

def create_smplx_model(model_path, gender='neutral'):
    return smplx.create(model_path, model_type='smplx', ext='npz', gender=gender, num_betas=11)


class SMPLXVisualizer:
    def __init__(self, faces):
        self.faces = faces
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=6.0)

        # Set up the camera
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4))
        self.light_node = self.scene.add(self.light, pose=np.eye(4))

        # Create a mesh node (we'll update its geometry later)
        self.mesh_node = None

        # Create an off-screen renderer
        self.renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=640)

        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.scene.set_pose(self.camera_node, camera_pose)

    def set_rotation(self, rotation_matrix):
        if self.scene.has_node(self.mesh_node):
            self.scene.set_pose(self.mesh_node, rotation_matrix)
            
    def update_mesh(self, vertices):
        new_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=vertices, faces=self.faces))
        if self.scene.has_node(self.mesh_node):
          self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(new_mesh)

    def render(self, save_path=None):
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        color, _ = self.renderer.render(self.scene) #,flags)
        image = Image.fromarray(color)

        if save_path:
            image.save(save_path, quality=95)

        return color
    

class SmlpxToPoseNormaliztor():

    def __init__(self, smplx_output, kpts_format = "coco"):
        self.kpts_format = kpts_format
        self.smplx_output = smplx_output
        self.joints = None
        self.vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()


    def transform_to_type_normalize_and_rotate_skeleton_facing_camera(self):
        self.get_pseudo_joints_by_format()
        skeleton_3d= self.rotate_skeleton_towards_camera_and_normalize_to_minus_one_to_one_by_format()

        return skeleton_3d


    def transform_to_type_normalize_and_rotate_model_facing_camera(self):
        self.get_pseudo_joints_by_format()
        vertices = self.rotate_model_towards_camera_and_normalize_to_minus_one_to_one_by_format()

        return vertices
        
    
    def get_pseudo_joints_by_format(self):
        """Extract Human3.6M joint positions from SMPLX output."""
        joints = self.smplx_output.joints.detach().cpu().numpy().squeeze()

        match self.kpts_format:
            case "h36m":
                # Map SMPLX joints to Human3.6M format (this mapping need to adjustment)
                joints = np.array([
                joints[1] + (joints[2] - joints[1])/2,   # Hip
                joints[2],   # RHip 1
                joints[5],   # RKnee
                joints[8],   # RFoot
                joints[1],   # LHip 4
                joints[4],   # LKnee
                joints[7],   # LFoot
                joints[9],   # Spine
                joints[12],  # Thorax
                joints[12] + (joints[15] - joints[12]) / 2,  # Neck (average of Thorax and Nose)
                joints[15],  # Nose
                joints[16],  # LShoulder 11
                joints[18],  # LElbow
                joints[20],  # LWrist
                joints[17],  # RShoulder 14
                joints[19],  # RElbow
                joints[21]   # RWrist
                ])
            case "coco":
                joints = np.array([
                joints[15],  # Nose
                joints[15],  # Left eye
                joints[15],  # Right eye
                joints[15],  # Left ear
                joints[15],  # Right ear
                joints[16],  # Left shoulder
                joints[17],  # Right shoulder
                joints[18],  # Left elbow
                joints[19],  # Right elbow
                joints[20],  # Left wrist
                joints[21],  # Right wrist
                joints[1],  # Left hip
                joints[2],  # Right hip
                joints[4],  # Left knee
                joints[5],  # Right knee
                joints[7],  # Left ankle
                joints[8],  # Right ankle
                ])
            case  _:
                raise Exception("Keypoint format is unknown")
            
        self.joints = joints


    def get_rotation_matrix_facing_camera_by_keypoints_type(self):

        final_rotation = []

        match self.kpts_format:
            case "coco":
                # Define key points for orientation
                hip_center = (self.joints[11] + self.joints[12]) / 2  # Mid-point between left and right hip
                chest = (self.joints[5] + self.joints[6]) / 2         # Mid-point between left and right shoulder
                right_hip = self.joints[12]                           # Right hip

                # Calculate the forward direction (from hip to chest)
                forward =  chest - hip_center
                forward = forward / np.linalg.norm(forward)

                # Calculate the right direction (from hip center to right hip)
                right = -(right_hip - hip_center)
                right = right / np.linalg.norm(right)

                # Calculate the up direction (cross product of forward and right)
                up = np.cross(forward, right)
                up = up / np.linalg.norm(up)

                # Recalculate the right vector to ensure orthogonality
                right = np.cross(up, forward)

                # Create the rotation matrix to face the camera
                rotation_matrix_camera = np.array([right, up, forward]).T

                # Create rotation matrix for 90 degrees around X-axis
                theta = np.radians(90)
                rotation_matrix_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]
                ])

                # Combine rotations
                final_rotation =  np.dot(rotation_matrix_camera ,rotation_matrix_x)

            case "h36m":
                # Define key points for orientation
                hip_center = self.joints[0]  
                chest = self.joints[8]        
                right_hip = self.joints[4]   

                # Calculate the forward direction (from hip to chest)
                forward =  chest - hip_center
                forward = forward / np.linalg.norm(forward)

                # Calculate the right direction (from hip center to right hip)
                right = right_hip - hip_center
                right = right / np.linalg.norm(right)

                # Calculate the up direction (cross product of forward and right)
                up = np.cross(forward, right)
                up = up / np.linalg.norm(up)

                # Recalculate the right vector to ensure orthogonality
                right = np.cross(up, forward)

                # Create the rotation matrix to face the camera
                rotation_matrix_camera = np.array([right, up, forward]).T

                # Create rotation matrix for 90 degrees around X-axis
                theta = np.radians(90)
                rotation_matrix_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]
                ])

                # Combine rotations
                final_rotation =  np.dot(rotation_matrix_camera ,rotation_matrix_x)
            case _:
                raise Exception("Keypoint format is unknown")

        return final_rotation

    def rotate_model_towards_camera_and_normalize_to_minus_one_to_one_by_format(self):
        
        normalized_vertices = []

        final_rotation = self.get_rotation_matrix_facing_camera_by_keypoints_type()

        match self.kpts_format:
            case "coco":
                # Define key points for orientation
                hip_center = (self.joints[11] + self.joints[12]) / 2  # Mid-point between left and right hip
                chest = (self.joints[5] + self.joints[6]) / 2         # Mid-point between left and right shoulder
                right_hip = self.joints[12] 

                # Apply the combined rotation to all joints
                rotated_joints = np.dot(self.vertices - hip_center, final_rotation)# + hip_center

                # Assuming 'rotated_joints' is already computed
                # Normalize 'rotated_joints' from -1 to 1
                max_abs = 0.7 * (np.linalg.norm((rotated_joints[11] + rotated_joints[12])/2) + np.linalg.norm((rotated_joints[5]+rotated_joints[6])/2)) #np.max(np.abs(rotated_joints))
                normalized_vertices = rotated_joints / max_abs

            case "h36m":
                # Define key points for orientation
                hip_center = self.joints[0]  
                chest = self.joints[8]        
                right_hip = self.joints[4]   

                # Apply the combined rotation to all joints
                rotated_joints = np.dot(self.vertices - hip_center, final_rotation)# + hip_center

                # Assuming 'rotated_joints' is already computed
                # Normalize 'rotated_joints' from -1 to 1
                max_abs = 0.7 * (np.linalg.norm((rotated_joints[4] + rotated_joints[1])/2) + np.linalg.norm((rotated_joints[11]+rotated_joints[14])/2)) #np.max(np.abs(rotated_joints))
                normalized_vertices = rotated_joints / max_abs
            case _:
                raise Exception("Keypoint format is unknown")

        return normalized_vertices


    def rotate_skeleton_towards_camera_and_normalize_to_minus_one_to_one_by_format(self):
        
        normalized_joints = []

        final_rotation = self.get_rotation_matrix_facing_camera_by_keypoints_type()

        match self.kpts_format:
            case "coco":
                # Define key points for orientation
                hip_center = (self.joints[11] + self.joints[12]) / 2  # Mid-point between left and right hip
                chest = (self.joints[5] + self.joints[6]) / 2         # Mid-point between left and right shoulder
                right_hip = self.joints[12] 

                # Apply the combined rotation to all joints
                rotated_joints = np.dot(self.joints - hip_center, final_rotation)# + hip_center

                # Assuming 'rotated_joints' is already computed
                # Normalize 'rotated_joints' from -1 to 1
                max_abs = 0.7 * (np.linalg.norm((rotated_joints[11] + rotated_joints[12])/2) + np.linalg.norm((rotated_joints[5]+rotated_joints[6])/2)) #np.max(np.abs(rotated_joints))
                normalized_joints = rotated_joints / max_abs

            case "h36m":
                # Define key points for orientation
                hip_center = self.joints[0]  
                chest = self.joints[8]        
                right_hip = self.joints[4]   

                # Apply the combined rotation to all joints
                rotated_joints = np.dot(self.joints - hip_center, final_rotation)# + hip_center

                # Assuming 'rotated_joints' is already computed
                # Normalize 'rotated_joints' from -1 to 1
                max_abs = 0.7 * (np.linalg.norm((rotated_joints[4] + rotated_joints[1])/2) + np.linalg.norm((rotated_joints[11]+rotated_joints[14])/2)) #np.max(np.abs(rotated_joints))
                normalized_joints = rotated_joints / max_abs
            case _:
                raise Exception("Keypoint format is unknown")

        return normalized_joints

    

# Function to load processed folders from file
def load_processed_folders(file_path='processed_folders'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            processed_folders = pickle.load(file)
        print(f"Loaded {len(processed_folders)} processed folders from file.")
        for ld_directory in processed_folders:
            print(ld_directory)
    else:
        processed_folders = []
        print("No processed folders file found. Starting fresh.")
    return processed_folders

# Function to save processed folders to file
def save_processed_folders(processed_folders, file_path='processed_folders'):
    with open(file_path, 'wb') as file:
        pickle.dump(processed_folders, file)
    print(f"Saved {len(processed_folders)} processed folders to file.")

import os
import numpy as np
import pickle
from tqdm import tqdm

def main():
    processed_folders_holder_name = "processed_folders_3d_motion"
    processed_folders = load_processed_folders(processed_folders_holder_name)

    # Start the virtual display
    display = Display(visible=0, size=(640, 640))
    display.start()

    detector = YOLODetector()

    final_normalized_skeleton_array = []
    
    final_res_path = '/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed_3d_keypoints_movement'
    model_path = '/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/bedlam_data/body_models/smplx/models/'
    directory_path = "/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/bedlam_data/smplx_gt/neutral_ground_truth_motioninfo"

    # Set the number of frames to skip
    frames_to_skip = 20
    next_frame_offset = 3 

    # Get the total number of collection folders
    total_collections = sum(1 for item in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, item)))

    # Main progress bar for collections
    for collection_folder_name in tqdm(sorted(os.listdir(directory_path)), total=total_collections, desc="Processing collections"):
        collection_folder_path = os.path.join(directory_path, collection_folder_name)
        if os.path.isdir(collection_folder_path) and collection_folder_path not in processed_folders:
            print(f"\nProcessing {collection_folder_path}")
            
            # Get the total number of motion folders in this collection
            total_motions = sum(1 for item in os.listdir(collection_folder_path) if os.path.isdir(os.path.join(collection_folder_path, item)))
            
            # Progress bar for motion folders within each collection
            for motion_folder_name in tqdm(sorted(os.listdir(collection_folder_path)), total=total_motions, desc="Processing motions", leave=False):
                motion_folder_path = os.path.join(collection_folder_path, motion_folder_name)
                if os.path.isdir(motion_folder_path):
                    motion_path = os.path.join(motion_folder_path, "motion_seq.npz")
                    
                    # Load motion sequence
                    motion_data = load_motion_sequence(motion_path)

                    # Get number of frames
                    num_frames = motion_data['poses'].shape[0]

                    # Ensure betas has the correct shape (should be 2D)
                    betas = motion_data['betas'].unsqueeze(0) if motion_data['betas'].dim() == 1 else motion_data['betas']

                    # Create SMPLX model
                    smplx_model = create_smplx_model(model_path)
                    
                    # Create visualizer
                    visualizer = SMPLXVisualizer(smplx_model.faces)
                    
                    # Get frames data
                    for frame in range(0, num_frames, frames_to_skip):
                        current_pose = process_frame(smplx_model, motion_data, frame, betas)
                        
                        # Process the next frame if it exists
                        next_frame = frame + next_frame_offset
                        if next_frame < num_frames:
                            next_pose = process_frame(smplx_model, motion_data, next_frame, betas)
                            # Combine current and next pose
                            combined_pose = np.concatenate([current_pose, next_pose], axis=1)
                            final_normalized_skeleton_array.append(combined_pose)

                            # fig1 = rotate_project_draw_two_3d_skeleton(current_pose, next_pose, y_rotation=90)
                            # fig1.savefig("pose_3d_current_and_next.jpg")

        # # Update processed folders after each collection
        # processed_folders.append(collection_folder_path)
        # save_processed_folders(processed_folders, processed_folders_holder_name)

    # Convert to numpy array
    final_normalized_skeleton_array = np.array(final_normalized_skeleton_array)

    # Save the results
    with open(os.path.join(final_res_path, '3d_skeletons_dataset.pkl'), 'wb') as f:
        pickle.dump(final_normalized_skeleton_array, f)

    # Stop the virtual display
    display.stop()

def process_frame(smplx_model, motion_data, frame, betas):
    poses = motion_data['poses'][frame:frame+1]
    trans = motion_data['trans'][frame:frame+1]

    # Ensure all inputs have the correct shape
    poses = poses.unsqueeze(0) if poses.dim() == 1 else poses
    trans = trans.unsqueeze(0) if trans.dim() == 1 else trans

    # Forward pass
    output = smplx_model(
        betas=betas,
        body_pose=poses[:, 3:66],
        global_orient=poses[:, :3],
        transl=trans
    )

    smplx_to_pose_normalizator = SmlpxToPoseNormaliztor(output, kpts_format="coco")
    rotated_joints = smplx_to_pose_normalizator.transform_to_type_normalize_and_rotate_skeleton_facing_camera()
    
    return rotated_joints

if __name__ == "__main__":
    main()