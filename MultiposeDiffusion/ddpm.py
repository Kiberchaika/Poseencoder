import torch
import torch.nn as nn

import gin
import numpy as np

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

@gin.configurable
class MultiposeDiffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, n_feat=256):
        super(MultiposeDiffusion, self).__init__()
        self.nn_model = nn_model(in_features=17*3, n_feat=n_feat).to(device)
        self.n_feat = n_feat
        
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.betas = betas

    
    def save(self, path, epoch, optimizer, train_loss, val_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, path)

    @classmethod
    def load(cls, config_path, ckpt_path, device):
        # Load the config
        gin.parse_config_file(config_path)

        # Get parameters from gin config
        n_T = gin.query_parameter('DDPM.n_T')
        n_feat = gin.query_parameter('DDPM.n_feat')
        betas = gin.query_parameter('DDPM.betas')

        # Create the model
        model = cls(ContextPoseUnet, betas, n_T, device, n_feat=n_feat)
        
        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint


    def get_alphas_cumprod(self):
        return self.alphabar_t

    def get_model(self):
        return self.nn_model

    def reshape_conditioning(self, x):
        # x shape: (batch_size, 2, 17, 2)
        batch_size, _, joints, _ = x.shape
        
        # First, we'll permute the dimensions to bring the ones we want to combine together
        x = x.permute(0, 2, 1, 3)
        # Now x shape: (batch_size, 17, 2, 2)
        
        # Then, we'll reshape to combine the last two dimensions
        x = x.reshape(batch_size, joints, 4)
        # Now x shape: (batch_size, 17, 4)
        
        return x

    def forward(self, x, c):
        # print(f"DDPM forward - x shape: {x.shape}, c shape: {c.shape}")
        
        c = self.reshape_conditioning(c)

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
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        
        if conditioning is None:
            conditioning = torch.zeros(n_sample, 17, 4).to(device)  # Default conditioning if none provided
        else:
            conditioning = self.reshape_conditioning(conditioning)
        
        # don't drop context at test time
        context_mask = torch.zeros(n_sample).to(device)

        # prepare context_mask for doubling
        context_mask_double = torch.cat([context_mask, torch.ones_like(context_mask)], dim=0)

        x_i_store = []  # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.full((n_sample,), i / self.n_T, device=device)

            # double batch
            x_i_double = torch.cat([x_i, x_i], dim=0)
            conditioning_double = torch.cat([conditioning, conditioning], dim=0)
            t_is_double = t_is.repeat(2)

            # split predictions and compute weighting
            eps = self.nn_model(x_i_double, conditioning_double, t_is_double, context_mask_double)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2

            # Update x_i using the combined eps
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store