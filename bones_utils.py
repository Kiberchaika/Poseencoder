import numpy as np
import cv2

BONES_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (0, 5), (0, 6)
]

BONES_H36M = [
    (0, 1), (1, 2), (2, 3),  # Right leg
    (0, 4), (4, 5), (5, 6),  # Left leg
    (0, 7), (7, 8), (8, 9),  # Spine and neck
    (9, 10),                 # Neck to Nose (head)
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16)   # Right arm
]

BONES_MEDIAPIPE = [
    (11, 12),  # shoulder_line
    (23, 24),  # waist_line
    (11, 23),  # left_shoulder_waist
    (12, 24),  # right_shoulder_waist
    (24, 26),  # right_thigh
    (23, 25),  # left_thigh
    (26, 28),  # right_leg
    (25, 27),  # left_leg
    (14, 16),  # right_forearm
    (13, 15),  # left_forearm
    (12, 14),  # right_bicep
    (11, 13)   # left_bicep
]

# Produce angular representation of a pose
def calculate_angles(self, points, body_indices):
    angles = []
    
    for bone in BONES_COCO:
        if bone[0] in body_indices and bone[1] in body_indices:
            index_1 = body_indices.index(bone[0])
            index_2 = body_indices.index(bone[1])
            v1 = points[index_1]
            v2 = points[index_2]
            angle = self.calculate_angle_between_vectors(v1, v2)
            angles.append(angle)

    return angles   

# Function to draw pose on image
def draw_pose_2d_cv(image, poses, type, conf=0.2):
    for pose in poses:
        for x, y, c in pose:
            if c > conf:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        for (i, j) in type:
            if pose[i][2] > conf and pose[j][2] > conf:
                cv2.line(image, (int(pose[i][0]), int(pose[i][1])), (int(pose[j][0]), int(pose[j][1])), (0, 255, 0), 2)

# Function to convert COCO keypoints to Human3.6M keypoints
def coco_to_h36m(coco_keypoints):
    """
    Convert COCO keypoints to Human3.6M keypoints.

    :param coco_keypoints: List of 17 COCO keypoints (x, y) coordinates.
    :return: List of 17 Human3.6M keypoints (x, y) coordinates.
    """
    # Define mapping from COCO to Human3.6M
    coco_to_h36m_mapping = {
        0: 9,    # Nose -> Nose
        5: 11,   # LShoulder -> L_Shoulder
        6: 14,   # RShoulder -> R_Shoulder
        7: 12,   # LElbow -> L_Elbow
        8: 15,   # RElbow -> R_Elbow
        9: 13,   # LWrist -> L_Wrist
        10: 16,  # RWrist -> R_Wrist
        11: 4,   # LHip -> L_HIP
        12: 1,   # RHip -> R_HIP
        13: 5,   # LKnee -> L_Knee
        14: 2,   # Rknee -> R_Knee
        15: 6,   # LAnkle -> L_Foot
        16: 3    # RAnkle -> R_Foot
    }

    # Initialize Human3.6M keypoints with None
    h36m_keypoints = [None] * 17

    # Convert COCO keypoints to Human3.6M using the mapping
    for coco_index, h36m_index in coco_to_h36m_mapping.items():
        h36m_keypoints[h36m_index] = coco_keypoints[coco_index]

    # Calculate Pelvis as the midpoint between LHip and RHip
    pelvis_x = (coco_keypoints[11][0] + coco_keypoints[12][0]) / 2
    pelvis_y = (coco_keypoints[11][1] + coco_keypoints[12][1]) / 2
    h36m_keypoints[0] = (pelvis_x, pelvis_y)

    # Calculate Spine as the midpoint between LShoulder and RShoulder
    spine_x = (coco_keypoints[5][0] + coco_keypoints[6][0]) / 2
    spine_y = (coco_keypoints[5][1] + coco_keypoints[6][1]) / 2
    h36m_keypoints[7] = (spine_x, spine_y)

    # Calculate Thorax as the midpoint between Spine and Nose
    thorax_x = (h36m_keypoints[7][0] + coco_keypoints[0][0]) / 2
    thorax_y = (h36m_keypoints[7][1] + coco_keypoints[0][1]) / 2
    h36m_keypoints[8] = (thorax_x, thorax_y)

    # Set Head as the Nose
    h36m_keypoints[10] = coco_keypoints[0]

    return np.array(h36m_keypoints)

def posetrack18_to_h36m(posetrack18_keypoints):
    """
    Convert PoseTrack18 keypoints to Human3.6M keypoints.

    :param posetrack18_keypoints: List of 15 PoseTrack18 keypoints (x, y) coordinates.
    :return: List of 17 Human3.6M keypoints (x, y) coordinates.
    """
    # Define mapping from PoseTrack18 to Human3.6M
    posetrack18_to_h36m_mapping = {
        0: 9,    # Nose -> Nose
        1: 8,    
        2: 10,   # Head Bottom -> Head
        5: 11,   # Left Shoulder -> L_Shoulder
        6: 14,   # Right Shoulder -> R_Shoulder
        7: 12,   # Left Elbow -> L_Elbow
        8: 15,   # Right Elbow -> R_Elbow
        9: 13,   # Left Wrist -> L_Wrist
        10: 16,   # Right Wrist -> R_Wrist
        11: 4,    # Left Hip -> L_Hip
        12: 1,   # Right Hip -> R_Hip
        13: 5,   # Left Knee -> L_Knee
        14: 2,   # Right Knee -> R_Knee
        15: 6,   # Left Ankle -> L_Foot
        16: 3    # Right Ankle -> R_Foot
    }

    # Initialize Human3.6M keypoints with None
    h36m_keypoints = [None] * 17

    # Convert PoseTrack18 keypoints to Human3.6M using the mapping
    for posetrack18_index, h36m_index in posetrack18_to_h36m_mapping.items():
        h36m_keypoints[h36m_index] = posetrack18_keypoints[posetrack18_index]

    # Calculate Pelvis as the midpoint between Left Hip and Right Hip
    pelvis_x = (posetrack18_keypoints[11][0] + posetrack18_keypoints[12][0]) / 2
    pelvis_y = (posetrack18_keypoints[11][1] + posetrack18_keypoints[12][1]) / 2
    h36m_keypoints[0] = (pelvis_x, pelvis_y)
    
    # Calculate Spine 
    spine_x = (posetrack18_keypoints[1][0] + pelvis_x) / 2
    spine_y = (posetrack18_keypoints[1][1] + pelvis_y) / 2
    h36m_keypoints[7] = (spine_x, spine_y)
    
    return np.array(h36m_keypoints)
