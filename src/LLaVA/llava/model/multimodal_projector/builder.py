import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class ReshapeLastTwoDim(nn.Module):
    def __init__(self):
        super(ReshapeLastTwoDim, self).__init__()

    def forward(self, x):
        return x.permute(0,2,1)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)


    mlp_downs_gelu_match = re.match(r'^mlp(\d+)x_gelu_downs(\d+)x$', projector_type)
    if mlp_downs_gelu_match:
        mlp_depth = int(mlp_downs_gelu_match.group(1))
        avg_stride = int(mlp_downs_gelu_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        
        modules.append(ReshapeLastTwoDim())
        modules.append(nn.AvgPool1d(kernel_size=avg_stride))
        modules.append(ReshapeLastTwoDim())
        
        return nn.Sequential(*modules)
    

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    resmlp_gelu_match = re.match(r'^resmlp(\d+)x_gelu$', projector_type)
    if resmlp_gelu_match:
        mlp_depth = int(resmlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ResidualBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        return x + self.gelu(self.linear(x))