import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import pickle
from bones_utils import BONES_COCO

def load_3d_keypoints_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        print(f"loaded {len(dataset)} skeletons")
        return dataset

# Load the skeleton data
skeletons = load_3d_keypoints_dataset("/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed_3d_keypoints/3d_skeletons_dataset.pkl")

def rotate_skeleton(skeleton, x_angle, y_angle, z_angle):
    def rotation_matrix(axis, angle):
        angle = np.radians(angle)
        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif axis == 'z':
            return np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])

    rot_x = rotation_matrix('x', x_angle)
    rot_y = rotation_matrix('y', y_angle)
    rot_z = rotation_matrix('z', z_angle)
    
    rotated_skeleton = np.dot(skeleton, rot_y)
    rotated_skeleton = np.dot(rotated_skeleton, rot_x)
    rotated_skeleton = np.dot(rotated_skeleton, rot_z)
    
    return rotated_skeleton

def perspective_projection(skeleton, fov, aspect_ratio, near, far, camera_distance):
    fov_rad = np.radians(fov)

    # Calculate the projection matrix
    f = 1 / np.tan(fov_rad / 2)
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, 1, 0]
    ])

    # Apply projection to each point
    projected_skeleton = []
    for point in skeleton:
        # Move the point away from the camera
        point_homogeneous = np.append(point, 1) * np.array([1, 1, -1, 1])  # Flip z-axis
        point_homogeneous[2] += camera_distance  # Move away from the camera
        
        projected_point = np.dot(projection_matrix, point_homogeneous)
        projected_point = projected_point[:3] / projected_point[3]  # Perspective divide
        projected_skeleton.append(projected_point[:2])  # Only keep x and y

    return np.array(projected_skeleton)

def randomize_limbs(skeleton, random_left, random_right):
    # Define indices for left and right limbs
    left_limb_indices = [7, 9, 13, 15]  # Left arm and leg
    right_limb_indices = [8, 10, 14, 16]  # Right arm and leg
    
    if random_left:
        random_skeleton = skeletons[np.random.randint(len(skeletons))]
        for idx in left_limb_indices:
            skeleton[idx] = random_skeleton[idx].copy()
    
    if random_right:
        random_skeleton = skeletons[np.random.randint(len(skeletons))]
        for idx in right_limb_indices:
            skeleton[idx] = random_skeleton[idx].copy()
    
    return skeleton

def rotate_project_draw_3d_skeleton(skeleton, 
                     x_rotation = 0, 
                     y_rotation = 0, 
                     z_rotation = 0, 
                     fov = 60, 
                     aspect_ratio = 1.0, 
                     near = 0.1, 
                     far = 100.0, 
                     camera_distance = 5.0):
    rotated_skeleton = rotate_skeleton(skeleton, x_rotation, y_rotation, z_rotation)
    projected_skeleton = perspective_projection(rotated_skeleton, fov, aspect_ratio, near, far, camera_distance)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot bones
    for start, end in BONES_COCO:
        ax.plot([projected_skeleton[start, 0], projected_skeleton[end, 0]],
                [projected_skeleton[start, 1], projected_skeleton[end, 1]], 'bo-', markersize=5, linewidth=2)
    
    # Annotate joints
    for j in range(projected_skeleton.shape[0]):
        ax.text(projected_skeleton[j, 0], projected_skeleton[j, 1], str(j), fontsize=8, ha='right')
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Skeleton \nFOV: {fov}°, Aspect Ratio: {aspect_ratio:.2f}\n"
                 f"Near: {near:.2f}, Far: {far:.2f}, Camera Distance: {camera_distance:.2f}\n")
    
    # Set fixed bounds for x and y axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)  # Invert y-axis
    
    plt.close(fig)  # Close the figure to prevent display
    
    return fig

def rotate_project_draw_two_3d_skeleton(
                     skeleton1, 
                     skeleton2, 
                     x_rotation = 0, 
                     y_rotation = 0, 
                     z_rotation = 0, 
                     fov = 60, 
                     aspect_ratio = 1.0, 
                     near = 0.1, 
                     far = 100.0, 
                     camera_distance = 5.0):
    
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Skeleton \nFOV: {fov}°, Aspect Ratio: {aspect_ratio:.2f}\n"
                 f"Near: {near:.2f}, Far: {far:.2f}, Camera Distance: {camera_distance:.2f}\n")
    
    # Set fixed bounds for x and y axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)  # Invert y-axis

    skeletons = [skeleton1, skeleton2]
    colors = ["green", "red"]

    for i in range(0, len(skeletons)):
        rotated_skeleton = rotate_skeleton(skeletons[i], x_rotation, y_rotation, z_rotation)
        projected_skeleton = perspective_projection(rotated_skeleton, fov, aspect_ratio, near, far, camera_distance)

        # Plot bones
        for start, end in BONES_COCO:
            ax.plot([projected_skeleton[start, 0], projected_skeleton[end, 0]],
                    [projected_skeleton[start, 1], projected_skeleton[end, 1]], 'bo-', markersize=5, linewidth=2, color = colors[i])
        
        # Annotate joints
        for j in range(projected_skeleton.shape[0]):
            ax.text(projected_skeleton[j, 0], projected_skeleton[j, 1], str(j), fontsize=8, ha='right', color = colors[i])    
    
    plt.close(fig)  # Close the figure to prevent display
    
    return fig


def visualize_skeleton(skeleton_index, x_rotation, y_rotation, z_rotation, fov, aspect_ratio, near, far, camera_distance, random_left, random_right):
    skeleton = skeletons[skeleton_index].copy()  # Make a copy to avoid modifying the original
    skeleton = randomize_limbs(skeleton, random_left, random_right)
    
    fig = rotate_project_draw_3d_skeleton(skeleton, x_rotation, y_rotation, z_rotation, fov, aspect_ratio, near, far, camera_distance)

    return fig

if __name__ == "__main__":

    # Create Gradio interface
    iface = gr.Interface(
        fn=visualize_skeleton,
        inputs=[
            gr.Slider(0, len(skeletons)-1, step=1, label="Skeleton Index"),
            gr.Slider(-180, 180, step=1, value=0, label="X Rotation"),
            gr.Slider(-180, 180, step=1, value=0, label="Y Rotation"),
            gr.Slider(-180, 180, step=1, value=0, label="Z Rotation"),
            gr.Slider(30, 120, step=1, value=60, label="Field of View (degrees)"),
            gr.Slider(0.5, 2.0, step=0.1, value=1.0, label="Aspect Ratio"),
            gr.Slider(0.1, 10.0, step=0.1, value=0.1, label="Near Plane"),
            gr.Slider(10.0, 1000.0, step=10.0, value=100.0, label="Far Plane"),
            gr.Slider(1.0, 20.0, step=0.1, value=5.0, label="Camera Distance"),
            gr.Checkbox(label="Random Left Limbs"),
            gr.Checkbox(label="Random Right Limbs")
        ],
        outputs="plot",
        title="Advanced 3D Skeleton Perspective Viewer with Random Limbs",
        description="Select a skeleton, adjust rotation, FOV, and projection parameters. Optionally randomize left or right limbs.",
    )

    # Launch the interface
    iface.launch(share=True)