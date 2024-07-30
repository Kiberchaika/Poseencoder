import torch
import torch.nn as nn

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
