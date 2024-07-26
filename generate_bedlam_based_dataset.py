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

HUMAN_MODEL_NAMES = ["male", "female", "neutral"]

'''
def get_coco_joints(smplx_output):
    """Extract COCO joint positions from SMPLX output."""
    joints = smplx_output.joints.detach().cpu().numpy().squeeze()

    # SMPLX to COCO joint mapping
    smplx_to_coco = [
        0,  # nose
        15,  # left_eye
        16,  # right_eye
        17,  # left_ear
        18,  # right_ear
        2,   # left_shoulder
        5,   # right_shoulder
        3,   # left_elbow
        6,   # right_elbow
        4,   # left_wrist
        7,   # right_wrist
        9,   # left_hip
        12,  # right_hip
        10,   # left_knee
        13,   # right_knee
        11,   # left_ankle
        14   # right_ankle
    ]
    

    coco_joints = joints[smplx_to_coco]
    return coco_joints
'''

#'''
def get_pseudo_h36m_joints(smplx_output):
    """Extract Human3.6M joint positions from SMPLX output."""
    joints = smplx_output.joints.detach().cpu().numpy().squeeze()
    
    # Map SMPLX joints to Human3.6M format (this mapping need to adjustment)
    h36m_joints = np.array([
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
    return h36m_joints
#'''

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

'''
def rotate_coco_model_towards_camera_and_normalize_to_minus_one_to_one(coco_joints, vert):
    # Define key points for orientation
    hip_center = (coco_joints[11] + coco_joints[12]) / 2  # Mid-point between left and right hip
    chest = (coco_joints[5] + coco_joints[6]) / 2         # Mid-point between left and right shoulder
    right_hip = coco_joints[12]                           # Right hip

    # Calculate the forward direction (from hip to chest)
    forward = chest - hip_center
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
    final_rotation = np.dot(rotation_matrix_camera, rotation_matrix_x)

    # Apply the combined rotation to all joints
    rotated_joints = np.dot(vert - hip_center, final_rotation)# + hip_center

    # Assuming 'rotated_joints' is already computed
    # Normalize 'rotated_joints' from -1 to 1
    max_abs = np.max(np.abs(rotated_joints))
    normalized_joints = rotated_joints / max_abs

    return normalized_joints
'''

def rotate_h36m_model_towards_camera_and_normalize_to_minus_one_to_one(h36m_joints, vert):
    # Define key points for orientation
    hip_center = h36m_joints[0]  
    chest = h36m_joints[8]        
    right_hip = h36m_joints[4]   

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

    # Apply the combined rotation to all joints
    rotated_joints = np.dot(vert - hip_center, final_rotation)# + hip_center

    # Assuming 'rotated_joints' is already computed
    # Normalize 'rotated_joints' from -1 to 1
    max_abs = 0.7 * (np.linalg.norm((rotated_joints[4] + rotated_joints[1])/2) + np.linalg.norm((rotated_joints[11]+rotated_joints[14])/2)) #np.max(np.abs(rotated_joints))
    normalized_joints = rotated_joints / max_abs

    return normalized_joints


def apply_transform_to_vertices(vertices, transform_matrix):
    # Add a homogeneous coordinate (1) to each vertex
    homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    
    # Apply the transformation
    transformed_vertices = np.dot(homogeneous_vertices, transform_matrix.T)
    
    # Remove the homogeneous coordinate
    return transformed_vertices[:, :3]

# Function to draw circles and lines on a numpy array image based on poses
def draw_pose_2d(img, poses, type, conf=0.2):
    # Iterate over each pose in the poses list
    for pose in poses:
        # Draw circles for each joint in the pose
        for i, (x, y, c) in enumerate(pose):
            # Check if the confidence of the joint is greater than the threshold
            if c > conf:
                # Draw a circle at the joint position
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1)

        # Draw lines connecting the joints as specified in the type list
        for (i, j) in type:
            # Check if the confidence of both joints is greater than the threshold
            if pose[i][2] > conf and pose[j][2] > conf:
                # Draw a line connecting the two joints
                cv2.line(img, (int(pose[i][0]), int(pose[i][1])), (int(pose[j][0]), int(pose[j][1])), (0, 255, 255), 2)

        cv2.circle(img, (int(pose[2][0]), int(pose[2][1])), 6, (0, 255, 0), -1)
        cv2.circle(img, (int(pose[16][0]), int(pose[16][1])),6 , (0, 255, 0), -1)
        cv2.circle(img, (int(pose[5][0]), int(pose[5][1])),6 , (0, 255, 0), -1)

    return img

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

def main():

    processed_folders = load_processed_folders()

    # Start the virtual display
    display = Display(visible=0, size=(640, 640))
    display.start()

    detector = YOLODetector()

    # Paths
    
    final_res_path = '/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed_v2_keypoints_only'
    # final_res_path = '/home/k4/Projects/Poseencoder/data'
    

    model_path = '/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/bedlam_data/body_models/smplx/models/'
    directory_path = "/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/bedlam_data/smplx_gt/neutral_ground_truth_motioninfo"

    for collection_folder_name in sorted(os.listdir(directory_path)):
        collection_folder_path = os.path.join(directory_path, collection_folder_name)
        if os.path.isdir(collection_folder_path):
            print(collection_folder_path)
        

    for collection_folder_name in sorted(os.listdir(directory_path)):
        collection_folder_path = os.path.join(directory_path, collection_folder_name)
        if os.path.isdir(collection_folder_path) and collection_folder_path not in processed_folders:
            print(f"processing {collection_folder_path}")
            # Iterate over all folders (directories) in the specified directory
            for motion_folder_name in sorted(os.listdir(collection_folder_path)):
                motion_folder_path = os.path.join(collection_folder_path, motion_folder_name)
                if os.path.isdir(motion_folder_path):
                    path_for_ft_naming = os.path.join(*motion_folder_path.split('/')[-2:])
                    motion_path = os.path.join(motion_folder_path, "motion_seq.npz")
                    
                    # Load motion sequence
                    motion_data = load_motion_sequence(motion_path)

                    save_folder = os.path.join(final_res_path, path_for_ft_naming)

                    # Get number of frames
                    num_frames = motion_data['poses'].shape[0]

                    # Ensure betas has the correct shape (should be 2D)
                    betas = motion_data['betas'].unsqueeze(0) if motion_data['betas'].dim() == 1 else motion_data['betas']

                    detected_indices = []
                    all_keypoints = np.zeros((num_frames, 17, 3))

                    for human_model_name in HUMAN_MODEL_NAMES:
                        # Create SMPLX smplx_model
                        smplx_model = create_smplx_model(model_path, human_model_name)
                        
                        # Create visualizer
                        visualizer = SMPLXVisualizer(smplx_model.faces)
                        
                        # Get frames data
                        for frame in range(0, num_frames, 20):
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

                            # Update mesh and render
                            vertices = output.vertices.detach().cpu().numpy().squeeze()

                            # Get and update bones
                            h36m_joints = get_pseudo_h36m_joints(output)
                            vertices = rotate_h36m_model_towards_camera_and_normalize_to_minus_one_to_one(h36m_joints, vertices)
                            visualizer.update_mesh(vertices)

                            rendered_image = visualizer.render()
                            rendered_image = np.copy(rendered_image) # make it writable
                            #visualizer.render("model.jpg")

                            # yolo
                            pred = detector.predict(np.array([rendered_image]))

                            ## Для теста - дальше идёт код, рендерящий текущий скелет
                            
                            # img = np.zeros((640, 640, 3), dtype = "uint8")
                            # img = draw_pose_2d(img, pred[0], BONES_COCO)
                            # cv2.imwrite("1.jpg", img)

                            # image_save_folder = os.path.join(save_folder, f"{human_model_name}/images")

                            # if not os.path.exists(image_save_folder):
                            #     os.makedirs(image_save_folder)

                            # complete_save_path = os.path.join(image_save_folder, f'{frame}.jpg')
                            # cv2.imwrite(complete_save_path, rendered_image)

                            if len(pred[0]) != 0 and frame not in detected_indices:
                                # Save the keypoints for this frame
                                all_keypoints[frame] = pred[0][0]
                                detected_indices.append(frame)
                            
                    normalized_keypoints = []
                    for keypoints in all_keypoints:
                        normalized_keypoints.append(keypoints[:, :2])

                    np.save(os.path.join(save_folder, "all_keypoints.npy"), normalized_keypoints)
        
        processed_folders.append(collection_folder_path)
        save_processed_folders(processed_folders)

    # Stop the virtual display
    display.stop()

main()
