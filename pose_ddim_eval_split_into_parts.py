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
import subprocess

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DDIMSampler(nn.Module):
    def __init__(self, model, beta_start_end, n_T, device):
        super().__init__()
        self.model = torch.compile(model)
        # self.model = model
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
        for steps in [1,2,3,4,5,6,7,8,9, 10, 11, 12, 15, 20, 50, 100, 200, 500, 1000]:
            t = torch.linspace(n_T - 1, 0, steps, dtype=torch.long, device=device)
            self.time_steps[steps] = t

        # Pre-compute alpha values
        self.alpha_t = {}
        self.alpha_t_next = {}
        for steps, t in self.time_steps.items():
            self.alpha_t[steps] = self.alphas_cumprod[t]
            self.alpha_t_next[steps] = torch.cat([self.alphas_cumprod[t[1:]], torch.tensor([1.0], device=device)])

    # @torch.jit.script
    def single_step_after(self, x, eps, t, alpha_t, alpha_t_next, guide_w, eta, n_T, context_mask_double):        
        eps1, eps2 = eps.chunk(2)
        eps = (1 + guide_w) * eps1 - guide_w * eps2

        sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next))

        c1 = torch.sqrt(alpha_t_next / alpha_t)
        c2 = torch.sqrt(1 - alpha_t_next - sigma_t**2) - torch.sqrt((alpha_t_next * (1 - alpha_t)) / alpha_t)
        x_next = c1.unsqueeze(1) * x + c2.unsqueeze(1) * eps

        return x_next, sigma_t
    
    def single_step_before(self, x, conditioning, t, alpha_t, alpha_t_next, guide_w, eta, n_T, context_mask_double):
        batch_size = x.shape[0]
        
        x_double = torch.cat([x, x], dim=0)
        c_double = torch.cat([conditioning, conditioning], dim=0)
        t_double = t.repeat(batch_size * 2)
        
        eps = self.model(x_double, c_double, t_double.float() / n_T, context_mask_double)
        
        return eps

    @torch.no_grad()
    def sample(self, x_t, conditioning, guide_w=0.0, steps=50, eta=0.0):
        profile = {}
        batch_size = x_t.shape[0]
        x = x_t

        start = time.time()
        time_steps = self.time_steps[steps]
        alpha_t = self.alpha_t[steps].unsqueeze(1).expand(steps, batch_size).t()
        alpha_t_next = self.alpha_t_next[steps].unsqueeze(1).expand(steps, batch_size).t()
        profile['alpha_t_unsqueeze'] = (time.time() - start) * 1000

        start = time.time()
        context_mask = torch.zeros(batch_size, device=self.device)
        context_mask_double = torch.cat([context_mask, torch.ones_like(context_mask)], dim=0)
        profile['context_mask'] = (time.time() - start) * 1000

        start = time.time()
        for i in range(steps):
            t = time_steps[i]
            
            eps = self.single_step_before(x, conditioning, t, alpha_t[:, i], alpha_t_next[:, i], 
                                          guide_w, eta, self.n_T, context_mask_double)
            x, sigma_t = self.single_step_after(x, eps, t, alpha_t[:, i], alpha_t_next[:, i], 
                                          guide_w, eta, self.n_T, context_mask_double)

            if i < steps - 1:
                noise = torch.randn_like(x)
                x += sigma_t.unsqueeze(1) * noise

        profile['sampling'] = (time.time() - start) * 1000

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

def generate_samples_and_video(ddim_sampler, input_tensor, device, n_frames=15, n_sample=1, w=1.5, steps=10, eta=1.0):
    output_dir = 'ddim_output'
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for j in range(n_frames):
            # DDIM sampling
            start_time = time.time()
            x_gen = sample_ddim(ddim_sampler, n_sample, (17, 3), device, guide_w=w, conditioning=input_tensor[0].to(device), steps=steps, eta=eta)
            end_time = time.time()

            ddim_sampling_time = (end_time - start_time) * 1000 / n_sample
            print(f"Frame {j+1}/{n_frames} - Average DDIM sampling time per pose (w={w}): {ddim_sampling_time:.2f} ms")

            fig = visualize_3d_and_2d_skeletons(x_gen[0].cpu().numpy(), input_tensor[0].cpu().numpy())
            output_path = os.path.join(output_dir, f'ddim_frame_{j:04d}.png')
            fig.savefig(output_path)
            plt.close(fig)

    # Generate video using ffmpeg
    video_output = 'ddim_animation.mp4'
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '30',  # Adjust as needed
        '-i', os.path.join(output_dir, 'ddim_frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        video_output
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video generated successfully: {video_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")

# Set device
device = "cuda:1" if torch.cuda.is_available() else "cpu"
n_feat = 512
n_T = 50
ws_test = [0.0, 0.5, 2.0]

torch.set_float32_matmul_precision('high')

# Initialize model
ddpm = DDPM(nn_model=ContextPoseUnet(in_features=17*3, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm.to(device)

ddim_sampler = DDIMSampler(ddpm.nn_model, ddpm.betas, ddpm.n_T, device)


# Load checkpoint
# checkpoint = torch.load('/home/k4/Projects/Poseencoder/pose_ddpm_runs/batch32_steps50_feats256/model_199.pth', map_location=device)
# checkpoint = torch.load('/home/k4/Projects/Poseencoder/pose_ddpm_runs/batch128_steps25_feats512/model_199.pth', map_location=device)
checkpoint = torch.load('pose_ddpm_runs/run_20240730_100605/model_55.pth', map_location=device)
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

torch.inference_mode()

# Evaluation
ddpm.eval()

with torch.no_grad():
    n_sample = 1

    generate_samples_and_video(ddim_sampler, input_tensor, device, 30, w=1.0, steps=15)

    # w = 1.5
    # x_gen, x_gen_store = ddpm.sample(n_sample, (17, 3), device, guide_w=w, conditioning=input_tensor[0].to(device))
    # start_time = time.time()
    # x_t = torch.randn(n_sample, 17, 3).to(device)
    # x_gen = ddim_sampler.sample(x_t, n_steps=50, eta=0.0)  # You can adjust n_steps and eta
    # x_gen = ddim_sampler.ddim_sample_loop(shape, 50, guide_w=0.0, conditioning=x_2d_samples, n_steps=50, eta=0.0)
    # end_time = time.time()

    # for j in range(0, 15):
    #     # DDIM sampling
    #     start_time = time.time()
    #     # x_gen, x_gen_store = ddpm.sample(n_sample, (17, 3), device, guide_w=w, conditioning=input_tensor[0].to(device))
    #     x_gen = sample_ddim(ddim_sampler, n_sample, (17, 3), device, guide_w=w, conditioning=input_tensor[0].to(device), steps=10, eta=1.0)
    #     end_time = time.time()



    #     ddim_sampling_time = (end_time - start_time) * 1000 / n_sample
    #     print(f"Average DDIM sampling time per pose (w={w}): {ddim_sampling_time:.2f} ms")


    #     fig = visualize_3d_and_2d_skeletons(x_gen[0].cpu().numpy(), input_tensor[0].cpu().numpy())
    #     fig.savefig(os.path.join(f'ddim_test{j}.jpg'))
    #     plt.close(fig)

