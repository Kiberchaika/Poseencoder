import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bones_utils import BONES_COCO
import os
import math
import time

import argparse
import gin

import sys

sys.path.append("../")
from dataset_poses import PosesDataset

from visual_utils import visualize_3d_and_2d_skeletons, plot_3d_skeleton, plot_2d_skeleton
from ddpm import MultiposeDiffusion
from unet import ContextPoseUnet
    
def fig_to_tensor(fig):
    # Convert matplotlib figure to tensor for TensorBoard
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return ToTensor()(image)

@gin.configurable
def train_pose(n_epoch, batch_size, lrate, ckpt_path=None, run_name=None):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    save_model = True
    base_save_dir = './pose_ddpm_runs/'
    ws_test = [0.0, 0.5, 2.0]

    print(f"Running training with batchsize={batch_size}, lrate={lrate}, n_epoch={n_epoch}")

    # Create a directory for this run
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    save_dir = os.path.join(base_save_dir, run_name)
    suffix = 1
    while os.path.exists(save_dir):
        save_dir = os.path.join(base_save_dir, f"{run_name}_{suffix}")
        suffix += 1
    os.makedirs(save_dir, exist_ok=True)

    # Save the config
    with open(os.path.join(save_dir, 'config.gin'), 'w') as f:
        f.write(gin.config_str())

    if ckpt_path:
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.gin')
        ddpm, checkpoint = MultiposeDiffusion.load(config_path, ckpt_path, device)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        ddpm = MultiposeDiffusion(ContextPoseUnet, gin.query_parameter('MultiposeDiffusion.betas'), 
                    gin.query_parameter('MultiposeDiffusion.n_T'), device, 
                    n_feat=gin.query_parameter('MultiposeDiffusion.n_feat'))
        ddpm.to(device)
        start_epoch = 0

    # Training dataset and dataloader
    train_dataset = PosesDataset(how_many_2d_poses_to_generate=2, split='train', use_additional_augment=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Validation dataset and dataloader
    val_dataset = PosesDataset(how_many_2d_poses_to_generate=2, split='val', use_additional_augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    if ckpt_path:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=save_dir)

    for ep in range(start_epoch, n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        # Training loop
        pbar = tqdm(train_dataloader)
        train_loss_sum = 0
        for i, x in enumerate(pbar):
            optim.zero_grad()
            x_3d = x[0]['skeleton_3d'].to(device)
            x_2d = x[0]['skeletons_2d'].to(device)
            loss = ddpm(x_3d, x_2d)
            loss.backward()
            train_loss_sum += loss.item()
            pbar.set_description(f"train loss: {loss.item():.4f}")
            optim.step()

            # Log training loss
            writer.add_scalar('Loss/train', loss.item(), ep * len(train_dataloader) + i)

        avg_train_loss = train_loss_sum / len(train_dataloader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, ep)
        
        # Validation and sampling
        # Validation and sampling
        ddpm.eval()
        with torch.no_grad():
            val_loss_sum = 0
            val_samples = []
            n_sample = 4
            sample_interval = math.ceil(len(val_dataloader) / n_sample)
            
            for i, x in enumerate(val_dataloader):
                x_3d = x[0]['skeleton_3d'].to(device)
                x_2d = x[0]['skeletons_2d'].to(device)
                loss = ddpm(x_3d, x_2d)
                val_loss_sum += loss.item()
                
                # Collect samples at specific intervals
                if i % sample_interval == 0 and len(val_samples) < n_sample:
                    val_samples.append((x_3d[0], x_2d[0]))  # Append first item of the batch
            
            avg_val_loss = val_loss_sum / len(val_dataloader)
            writer.add_scalar('Loss/validation', avg_val_loss, ep)

            print(f"Epoch {ep} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Sample and visualize
            for w_i, w in enumerate(ws_test):
                x_2d_samples = torch.stack([sample[1] for sample in val_samples])
                x_3d_samples = torch.stack([sample[0] for sample in val_samples])
                
                # Measure sampling time
                start_time = time.time()
                x_gen, x_gen_store = ddpm.sample(n_sample, (17, 3), device, guide_w=w, conditioning=x_2d_samples)
                end_time = time.time()
                
                # Calculate and print the average time per sample
                sampling_time = (end_time - start_time) * 1000 / n_sample  # Convert to milliseconds
                print(f"Average sampling time per pose (w={w}): {sampling_time:.2f} ms")
                
                for i in range(n_sample):
                    fig = visualize_3d_and_2d_skeletons(x_gen[i].cpu().numpy(), 
                                                        x_2d_samples[i].cpu().numpy(),
                                                        x_3d_samples[i].cpu().numpy())
                    writer.add_figure(f'Generated_Pose_with_Conditioning/w_{w}_sample_{i}', fig, ep)
                    plt.close(fig)

        if save_model:
            model_save_path = os.path.join(save_dir, f"model_{ep}.pth")
            ddpm.save(model_save_path, ep, optim, avg_train_loss, avg_val_loss)
            print(f'Saved model at {model_save_path}')

    writer.close()
    print(f"Training complete. Outputs saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM for 3D pose generation")
    parser.add_argument("--config", type=str, default="default.gin", help="Path to the gin config file")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint file to resume training from", default=None)
    parser.add_argument("--name", type=str, help="Name for the run", default=None)
    parser.add_argument("--n_epoch", type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lrate", type=float, help="Learning rate")
    args = parser.parse_args()

    # Load the config
    gin.parse_config_file(args.config)
    
    # Override config with command line arguments if provided
    if args.n_epoch:
        gin.bind_parameter('train_pose.n_epoch', args.n_epoch)
    if args.batch_size:
        gin.bind_parameter('train_pose.batch_size', args.batch_size)
    if args.lrate:
        gin.bind_parameter('train_pose.lrate', args.lrate)

    # Get parameters from gin config
    n_epoch = gin.query_parameter('train_pose.n_epoch')
    batch_size = gin.query_parameter('train_pose.batch_size')
    lrate = gin.query_parameter('train_pose.lrate')

    train_pose(n_epoch, batch_size, lrate, ckpt_path=args.ckpt, run_name=args.name)