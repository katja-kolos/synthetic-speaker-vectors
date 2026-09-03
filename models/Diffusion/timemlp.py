import os
import math
import torch
import torch.nn as nn
import numpy as np
import einops
from einops import repeat

from .resnet import ResBlock, zero_module

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    # turn scalar timestep into a vector: smooth multi-scale encoding of time
    # timestep controls noise level
    # model behaves differently at early timesteps (almost noise) vs late timesteps (clear speaker vectors)
    # nearby timesteps should have similar representations
    # ...and (!) the model must be able to interpolate to unseen timesteps
    # solution: embeddings!

    # embedding idea: sinusoidal embeddings. 
    # Low-freq sinusoids capture coarse time info, high-freq sinusoids capture fine-grained differences
    # max_period controls the lowest frequency
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        # timesteps[:, None]: shape [B, 1]
        # freqs[None]: shape [1, half]
        # args: [B, half]
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) #[B, dim]
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim) # this is not recommended for diffusion
    return embedding

class ResMLPNet(nn.Module):
    # the main model with timestep embedding
    # note: this is not a proper UNet, just a stack of residual blocks 
    # with time embedding injected at each one

    def __init__(
        self,
        in_channels=256,
        time_embed_dim=256,
        model_channels=512,
        bottleneck_channels=512,
        out_channels=256,
        num_res_blocks=3,
        dropout=0,
        use_context=True,
        context_channels=256
    ):
        super().__init__()
        self.dim = out_channels

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                bottleneck_channels,
                time_embed_dim,
                dropout,
                use_context=use_context,
                context_channels=context_channels
            ))

        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(model_channels, eps=1e-6),
            nn.SiLU(),
            zero_module(nn.Linear(model_channels, out_channels, bias=True)),
        )

    def forward(self, x, timesteps=None, cond=None, y=None, **kwargs):
        # Inputs:
        #  x: tensor [N x C x ...] 
        #  timesteps: 1d batch
        #  context: conditioning throug cross-attention; in SEED-v2, x_cond is text embedding, we leave this for future work
        #  y: tensor of size [N] with labels
        # Returns:
        #  [N x C x ...]

        # print(f"ResMLPNet: x.shape: {x.shape}")
        x = x.squeeze()
        # print(f"ResMLPNet: x.squeeze().shape: {x.shape}")
        x = self.input_proj(x)
        # print(f"ResMLPNet: input_proj shape: {x.shape}")
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        # print(f"ResMLPNet: t_emb shape: {t_emb.shape}")
        emb = self.time_embed(t_emb)
        # print(f"ResMLPNet: emb shape: {emb.shape}")

        for block in self.res_blocks:
            # x: (B, 256), emb: (B, 128), cond: None
            x = block(x, emb, cond)
            print(f"ResMLPNet: x after resnet block shape: {x.shape}")

        out = self.out(x)
        print(f"ResMLPNet: out.shape: {out.shape}")
        return out