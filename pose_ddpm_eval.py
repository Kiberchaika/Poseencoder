import torch
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

from pose_ddpm_train import DDPM, ContextPoseUnet
from dataset_poses import PosesDataset
from bones_utils import BONES_COCO

def visualize_3d_and_2d_skeletons(skeleton_3d, skeletons_2d):
    fig = plt.figure(figsize=(15, 5))
    
    # 3D subplot
    ax_3d = fig.add_subplot(131, projection='3d')
    plot_3d_skeleton(ax_3d, skeleton_3d)
    ax_3d.set_title('Generated 3D Pose')
    
    # 2D subplots
    ax_2d_1 = fig.add_subplot(132)
    plot_2d_skeleton(ax_2d_1, skeletons_2d[0])
    ax_2d_1.set_title('2D Conditioning (View 1)')
    
    ax_2d_2 = fig.add_subplot(133)
    plot_2d_skeleton(ax_2d_2, skeletons_2d[1])
    ax_2d_2.set_title('2D Conditioning (View 2)')
    
    plt.tight_layout()
    return fig

def plot_3d_skeleton(ax, skeleton):
    for start, end in BONES_COCO:
        ax.plot([skeleton[start, 0], skeleton[end, 0]],
                [skeleton[start, 1], skeleton[end, 1]],
                [skeleton[start, 2], skeleton[end, 2]], 'bo-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def plot_2d_skeleton(ax, skeleton):
    for start, end in BONES_COCO:
        ax.plot([skeleton[start, 0], skeleton[end, 0]],
                [skeleton[start, 1], skeleton[end, 1]], 'ro-') 
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # Remove the invert_yaxis() call
    ax.set_aspect('equal', adjustable='box')  # Ensure correct aspect ratio


# Set device
device = "cuda:1" if torch.cuda.is_available() else "cpu"
n_feat = 256
n_T = 25
ws_test = [0.0, 0.5, 2.0]

# Initialize model
ddpm = DDPM(nn_model=ContextPoseUnet(in_features=17*3, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm.to(device)

# Load checkpoint
checkpoint = torch.load('pose_ddpm_runs/batchsize_16_n_t_25_2/model_99.pth', map_location=device)
ddpm.load_state_dict(checkpoint['model_state_dict']) # не работает загрузка

'''
# hack load
state_dict = OrderedDict()
for key in checkpoint.keys():
    if key.startswith('nn_model'):
        new_key = key.replace('nn_model.', '')
        state_dict[new_key] = checkpoint[key]

# Load the state_dict into the model
ddpm.load_state_dict(state_dict, strict=False)
'''

dataset = PosesDataset(how_many_2d_poses_to_generate=2, use_additional_augment=False, split='train')
item = dataset[1000]
input_tensor  = torch.tensor(item[0]['skeletons_2d'], dtype=torch.float32).unsqueeze(0)

# Evaluation
ddpm.eval()

with torch.no_grad():
    n_sample = 1
    w = 2.0
    x_gen, x_gen_store = ddpm.sample(n_sample, (17, 3), device, guide_w=w, conditioning=input_tensor[0].to(device))
    for i in range(n_sample):
        fig = visualize_3d_and_2d_skeletons(x_gen[i].cpu().numpy(), input_tensor[0].cpu().numpy())
        fig.savefig(os.path.join(f'ddpm_test.jpg'))
        plt.close(fig)

