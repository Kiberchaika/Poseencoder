import torch
import torch.nn as nn
from blocks import ResidualLinearBlock, PoseDown, PoseUp, EmbedFC


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
        batch_size = x.shape[0]
        
        x = x.view(batch_size, -1)  # Flatten the input
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        c = c.view(batch_size, -1)

        # mask out context if context_mask == 1
        context_mask = context_mask.view(-1, 1)
        context_mask = context_mask.repeat(1, c.shape[1])
        context_mask = (-1*(1-context_mask))
        c = c * context_mask
        
        cemb1 = self.contextembed1(c)
        temb1 = self.timeembed1(t.unsqueeze(1))
        cemb2 = self.contextembed2(c)
        temb2 = self.timeembed2(t.unsqueeze(1))

        up1 = self.up0(hiddenvec)
        up1 = up1 + cemb1 + temb1
        up2 = self.up1(up1, down2)
        up2 = up2 + cemb2 + temb2
        up3 = self.up2(up2, down1)
        out = self.out(up3)
        
        return out.view(batch_size, 17, 3)  # Reshape back to (batch_size, 17, 3)