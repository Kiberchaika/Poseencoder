from bones_utils import BONES_COCO
import matplotlib.pyplot as plt

def visualize_3d_and_2d_skeletons(skeleton_3d, skeletons_2d, ground_truth_3d):
    fig = plt.figure(figsize=(15, 5))
    
    # 3D subplot
    ax_3d = fig.add_subplot(131, projection='3d')
    plot_3d_skeleton(ax_3d, skeleton_3d, color='b', label='Generated')
    plot_3d_skeleton(ax_3d, ground_truth_3d, color='g', label='Ground Truth')
    ax_3d.set_title('3D Poses')
    ax_3d.legend()
    
    # 2D subplots
    ax_2d_1 = fig.add_subplot(132)
    plot_2d_skeleton(ax_2d_1, skeletons_2d[0])
    ax_2d_1.set_title('2D Conditioning (View 1)')
    
    ax_2d_2 = fig.add_subplot(133)
    plot_2d_skeleton(ax_2d_2, skeletons_2d[1])
    ax_2d_2.set_title('2D Conditioning (View 2)')
    
    plt.tight_layout()
    return fig

def plot_3d_skeleton(ax, skeleton, color='b', label=None):
    for start, end in BONES_COCO:
        ax.plot([skeleton[start, 0], skeleton[end, 0]],
                [skeleton[start, 2], skeleton[end, 2]],  # Swapped y and z
                [skeleton[start, 1], skeleton[end, 1]],  # Swapped y and z
                color=color)
    ax.set_xlabel('X')
    ax.set_zlabel('Y')  # Swapped y and z
    ax.set_ylabel('Z')  # Swapped y and z
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    if label:
        ax.plot([], [], [], color=color, label=label)

def plot_2d_skeleton(ax, skeleton):
    for start, end in BONES_COCO:
        ax.plot([skeleton[start, 0], skeleton[end, 0]],
                [skeleton[start, 1], skeleton[end, 1]], 'ro-') 
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')  # Ensure correct aspect ratio
