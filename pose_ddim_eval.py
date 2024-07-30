import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

from pose_ddpm_train import DDPM, ContextPoseUnet
from dataset_poses import PosesDataset
from bones_utils import BONES_COCO
import time

import numpy as np

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DDIMSampler(nn.Module):
    def __init__(self, model, beta_start_end, n_T, device):
        super().__init__()
        self.model = torch.compile(model)
        self.n_T = n_T
        self.device = device

        # Generate beta schedule
        beta_start, beta_end = beta_start_end
        self.betas = torch.linspace(beta_start, beta_end, n_T, dtype=torch.float32, device=device)
        
        # Calculate alphas and alphas_cumprod
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # Pre-compute time steps
        self.time_steps = {}
        for steps in [1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 200, 500, 1000]:  # Add or remove steps as needed
            t = torch.linspace(n_T - 1, 0, steps, dtype=torch.long, device=device)
            self.time_steps[steps] = t

        # Pre-compute alpha values
        self.alpha_t = {}
        self.alpha_t_next = {}
        for steps, t in self.time_steps.items():
            self.alpha_t[steps] = self.alphas_cumprod[t]
            self.alpha_t_next[steps] = torch.cat([self.alphas_cumprod[t[1:]], torch.tensor([1.0], device=device)])

    @torch.no_grad()
    def sample(self, x_t, conditioning, guide_w=0.0, steps=50, eta=0.0):
        profile = {}
        batch_size = x_t.shape[0]
        x = x_t

        start = time.time()
        # Use pre-computed time steps and alpha values
        time_steps = self.time_steps[steps]
        alpha_t = self.alpha_t[steps].unsqueeze(1).expand(steps, batch_size).t()
        alpha_t_next = self.alpha_t_next[steps].unsqueeze(1).expand(steps, batch_size).t()
        profile['alpha_t_unsqueeze'] = (time.time() - start) * 1000

        start = time.time()
        # Pre-compute context masks
        context_mask = torch.zeros(batch_size, device=self.device)
        context_mask_double = torch.cat([context_mask, torch.ones_like(context_mask)], dim=0)
        profile['context_mask'] = (time.time() - start) * 1000

        for i in range(steps):
            t = time_steps[i]
            
            start = time.time()
            # Double the batch for guided diffusion
            x_double = torch.cat([x, x], dim=0)
            c_double = torch.cat([conditioning, conditioning], dim=0)
            t_double = t.repeat(batch_size * 2)
            profile['double_batch'] = profile.get('double_batch', 0) + (time.time() - start) * 1000
            
            start = time.time()
            # Predict noise
            eps = self.model(x_double, c_double, t_double.float() / self.n_T, context_mask_double)
            profile['predict_noise'] = profile.get('predict_noise', 0) + (time.time() - start) * 1000
            
            start = time.time()
            # Apply guidance
            eps1, eps2 = eps.chunk(2)
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            profile['apply_guidance'] = profile.get('apply_guidance', 0) + (time.time() - start) * 1000

            start = time.time()
            # Compute sigma_t
            sigma_t = eta * torch.sqrt((1 - alpha_t_next[:, i]) / (1 - alpha_t[:, i]) * (1 - alpha_t[:, i] / alpha_t_next[:, i]))
            profile['compute_sigma_t'] = profile.get('compute_sigma_t', 0) + (time.time() - start) * 1000

            start = time.time()
            # Compute x_{t-1}
            c1 = torch.sqrt(alpha_t_next[:, i] / alpha_t[:, i])
            c2 = torch.sqrt(1 - alpha_t_next[:, i] - sigma_t**2) - torch.sqrt((alpha_t_next[:, i] * (1 - alpha_t[:, i])) / alpha_t[:, i])
            x_next = c1.unsqueeze(1) * x + c2.unsqueeze(1) * eps
            profile['compute_x_next'] = profile.get('compute_x_next', 0) + (time.time() - start) * 1000

            if i < steps - 1:
                start = time.time()
                noise = torch.randn_like(x)
                x_next += sigma_t.unsqueeze(1) * noise
                profile['add_noise'] = profile.get('add_noise', 0) + (time.time() - start) * 1000

            x = x_next

        return x, profile
    
# Usage
def sample_ddim(ddim, n_sample, size, device, guide_w=0.0, conditioning=None, steps=50, eta=0.0):
    x = torch.randn(n_sample, *size, device=device)
    sample, profile = ddim.sample(x, conditioning, guide_w, steps, eta)
    print("DDIM Sampling Profile:")
    for key, value in profile.items():
        print(f"{key}: {value:.2f} ms")
    
    return sample

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
n_feat = 512
n_T = 25
ws_test = [0.0, 0.5, 2.0]

# Initialize model
ddpm = DDPM(nn_model=ContextPoseUnet(in_features=17*3, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm.to(device)

ddim_sampler = DDIMSampler(ddpm.nn_model, ddpm.betas, ddpm.n_T, device)


# Load checkpoint
# checkpoint = torch.load('pose_ddpm_runs/run_20240730_095911/model_94.pth', map_location=device) # 50 steps
checkpoint = torch.load('pose_ddpm_runs/run_20240730_084456/model_299.pth', map_location=device) # 25 steps
ddpm.load_state_dict(checkpoint['model_state_dict']) # не работает загрузка

dataset = PosesDataset(how_many_2d_poses_to_generate=2, use_additional_augment=False, split='train')
conditioning_target = dataset[1000]
conditioning_input  = torch.tensor(conditioning_target[0]['skeletons_2d'], dtype=torch.float32).unsqueeze(0)

torch.inference_mode()

# Evaluation
ddpm.eval()

with torch.no_grad():
    n_sample = 1
    w = 1.5

    for j in range(0, 15):
        # DDIM sampling
        start_time = time.time()
        x_gen_ddim = sample_ddim(ddim_sampler, n_sample, (17, 3), device, guide_w=w, conditioning=conditioning_input[0].to(device), steps=10, eta=1.0)
        end_time = time.time()

        ddim_sampling_time = (end_time - start_time) * 1000 / n_sample
        print(f"Average DDIM sampling time per pose (w={w}): {ddim_sampling_time:.2f} ms")

        for i in range(n_sample):
            fig = visualize_3d_and_2d_skeletons(x_gen_ddim[i].cpu().numpy(), conditioning_input[0].cpu().numpy())
            fig.savefig(os.path.join(f'ddim_test{j}.jpg'))
            plt.close(fig)

