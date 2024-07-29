import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from tqdm import tqdm
from dataset_poses import PosesDataset
from pose_3d_perspective_projection import rotate_project_draw_3d_skeleton
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from datetime import datetime

class ResidualLinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_features = in_features == out_features
        self.is_res = is_res
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.linear1(x)
            x2 = self.linear2(x1)
            if self.same_features:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.linear1(x)
            x2 = self.linear2(x1)
            return x2

class PoseDown(nn.Module):
    def __init__(self, in_features, out_features):
        super(PoseDown, self).__init__()
        self.model = nn.Sequential(
            ResidualLinearBlock(in_features, out_features),
            nn.Linear(out_features, out_features)  # Removed the dimension reduction
        )

    def forward(self, x):
        return self.model(x)

class PoseUp(nn.Module):
    def __init__(self, in_features, out_features):
        super(PoseUp, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            ResidualLinearBlock(out_features, out_features),
            ResidualLinearBlock(out_features, out_features),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        # print(f"PoseUp input shape after concat: {x.shape}")
        x = self.model(x)
        # print(f"PoseUp output shape: {x.shape}")
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextPoseUnet(nn.Module):
    def __init__(self, in_features, n_feat = 256):
        super(ContextPoseUnet, self).__init__()
        self.in_features = in_features
        self.n_feat = n_feat

        self.init_conv = ResidualLinearBlock(in_features, n_feat, is_res=True)
        self.down1 = PoseDown(n_feat, n_feat)
        self.down2 = PoseDown(n_feat, n_feat)

        self.to_vec = nn.Sequential(nn.Linear(n_feat, n_feat), nn.GELU())

        self.timeembed1 = EmbedFC(1, n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(17*2*2, n_feat)
        self.contextembed2 = EmbedFC(17*2*2, n_feat)

        self.up0 = nn.Sequential(
            nn.Linear(n_feat, n_feat),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
        )

        self.up1 = PoseUp(n_feat, n_feat)
        self.up2 = PoseUp(n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Linear(n_feat, n_feat),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, self.in_features),
        )

    def forward(self, x, c, t, context_mask):
        # print(f"Input x shape: {x.shape}")
        
        x = x.view(x.shape[0], -1)  # Flatten the input
        # print(f"Flattened x shape: {x.shape}")
        
        x = self.init_conv(x)
        # print(f"After init_conv x shape: {x.shape}")
        
        down1 = self.down1(x)
        # print(f"down1 shape: {down1.shape}")
        
        down2 = self.down2(down1)
        # print(f"down2 shape: {down2.shape}")
        
        hiddenvec = self.to_vec(down2)
        # print(f"hiddenvec shape: {hiddenvec.shape}")

        # print(f"Input c shape: {c.shape}")
        
        c = c.view(c.shape[0], -1)
        # print(f"Flattened c shape: {c.shape}")

        # print(f"context_mask shape: {context_mask.shape}")
        
        context_mask = context_mask.view(-1, 1)
        # print(f"Reshaped context_mask shape: {context_mask.shape}")
        
        context_mask = context_mask.repeat(1, c.shape[1])
        # print(f"Repeated context_mask shape: {context_mask.shape}")
        
        context_mask = (-1*(1-context_mask))
        c = c * context_mask
        # print(f"Masked c shape: {c.shape}")
        
        cemb1 = self.contextembed1(c)
        # print(f"cemb1 shape: {cemb1.shape}")
        
        temb1 = self.timeembed1(t.unsqueeze(1))
        # print(f"temb1 shape: {temb1.shape}")
        
        cemb2 = self.contextembed2(c)
        # print(f"cemb2 shape: {cemb2.shape}")
        
        temb2 = self.timeembed2(t.unsqueeze(1))
        # print(f"temb2 shape: {temb2.shape}")

        up1 = self.up0(hiddenvec)
        # print(f"up1 shape (after up0): {up1.shape}")
        up1 = up1 + cemb1 + temb1
        # print(f"up1 shape (after addition): {up1.shape}")
        up2 = self.up1(up1, down2)
        # print(f"up2 shape (before addition): {up2.shape}")
        up2 = up2 + cemb2 + temb2
        # print(f"up2 shape (after addition): {up2.shape}")
        up3 = self.up2(up2, down1)
        # print(f"up3 shape: {up3.shape}")
        out = self.out(up3)
        # print(f"out shape before reshape: {out.shape}")
        
        out = out.view(-1, 17, 3)  # Reshape back to (batch_size, 17, 3)
        # print(f"Final out shape: {out.shape}")
        
        return out
    
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        # print(f"DDPM forward - x shape: {x.shape}, c shape: {c.shape}")
        
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        # print(f"_ts shape: {_ts.shape}")
        
        noise = torch.randn_like(x)
        # print(f"noise shape: {noise.shape}")

        x_t = (
            self.sqrtab[_ts, None, None] * x
            + self.sqrtmab[_ts, None, None] * noise
        )
        # print(f"x_t shape: {x_t.shape}")

        # Change this line to generate a 1D context_mask
        context_mask = torch.bernoulli(torch.zeros(x.shape[0]) + self.drop_prob).to(self.device)
        # print(f"context_mask shape: {context_mask.shape}")
        
        # Print shapes right before calling nn_model
        # print(f"Before nn_model - x_t: {x_t.shape}, c: {c.shape}, _ts: {_ts.shape}, context_mask: {context_mask.shape}")
        
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0, conditioning=None):
        x_i = torch.randn(n_sample, *size).to(device)
        
        if conditioning is None:
            conditioning = torch.zeros(n_sample, 17, 2, 2).to(device)  # Default conditioning if none provided
        
        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.full((n_sample,), i / self.n_T, device=device)
            
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            
            eps = self.nn_model(x_i, conditioning, t_is, torch.zeros(n_sample, device=device))
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

def fig_to_tensor(fig):
    # Convert matplotlib figure to tensor for TensorBoard
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return ToTensor()(image)

def train_pose():
    n_epoch = 100
    batch_size = 128
    n_T = 400
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    n_feat = 256
    lrate = 1e-4
    save_model = True
    base_save_dir = './data/diffusion_outputs_pose/'
    ws_test = [0.0, 0.5, 2.0]

    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    ddpm = DDPM(nn_model=ContextPoseUnet(in_features=17*3, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # Training dataset and dataloader
    train_dataset = PosesDataset(how_many_2d_poses_to_generate=2, split='train', use_additional_augment=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Validation dataset and dataloader
    val_dataset = PosesDataset(how_many_2d_poses_to_generate=2, split='val', use_additional_augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    # Initialize TensorBoard writer with the new save directory
    writer = SummaryWriter(log_dir=save_dir)

    for ep in range(n_epoch):
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
        
        # Validation loop
        ddpm.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for x in val_dataloader:
                x_3d = x[0]['skeleton_3d'].to(device)
                x_2d = x[0]['skeletons_2d'].to(device)
                loss = ddpm(x_3d, x_2d)
                val_loss_sum += loss.item()
            
            avg_val_loss = val_loss_sum / len(val_dataloader)
            writer.add_scalar('Loss/validation', avg_val_loss, ep)

            print(f"Epoch {ep} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Sample and visualize
            n_sample = 4
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (17, 3), device, guide_w=w, conditioning=x_2d[:n_sample])
                
                for i in range(n_sample):
                    fig = rotate_project_draw_3d_skeleton(x_gen[i].cpu().numpy())
                    writer.add_image(f'Generated_Pose/w_{w}_sample_{i}', fig_to_tensor(fig), ep)

        if save_model and ep == int(n_epoch-1):
            model_save_path = os.path.join(save_dir, f"model_{ep}.pth")
            torch.save(ddpm.state_dict(), model_save_path)
            print(f'Saved model at {model_save_path}')

    writer.close()
    print(f"Training complete. Outputs saved in {save_dir}")

if __name__ == "__main__":
    train_pose()