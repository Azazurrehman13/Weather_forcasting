"""
Torch side of the app: the SimVP architecture (verbatim from training) and a
loader that returns a met_core.Forecaster. All non-torch logic lives in met_core.py.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from met_core import Stats, Forecaster, C, INPUT_LEN, HORIZON, PAD_H, PAD_W, ORIG_H, ORIG_W

HIDDEN_WIDTH = 64


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class InceptionResBlock(nn.Module):
    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels // 2
        branch_ch = hidden // 3
        self.branch3 = nn.Conv2d(channels, branch_ch, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(channels, branch_ch, kernel_size=5, padding=2)
        self.branch7 = nn.Conv2d(channels, hidden - 2 * branch_ch, kernel_size=7, padding=3)
        self.norm1 = nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x):
        b = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        b = self.act(self.norm1(b))
        b = self.norm2(self.project(b))
        return self.act(x + b)


class SimVP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_width=64, translator_depth=6):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, hidden_width, stride=1)
        self.enc2 = ConvBlock(hidden_width, hidden_width, stride=2)
        self.enc3 = ConvBlock(hidden_width, hidden_width, stride=2)
        self.translator = nn.Sequential(*[InceptionResBlock(hidden_width) for _ in range(translator_depth)])
        self.up1 = nn.ConvTranspose2d(hidden_width, hidden_width, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(hidden_width * 2, hidden_width, stride=1)
        self.up2 = nn.ConvTranspose2d(hidden_width, hidden_width, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(hidden_width, hidden_width, stride=1)
        self.head = nn.Conv2d(hidden_width, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        t = self.translator(e3)
        d1 = self.up1(t)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(d2)
        return self.head(d2)


def load_forecaster(checkpoint_path, stats_dir, device="cpu"):
    """Build the model, load weights + normalization stats, return a Forecaster."""
    stats = Stats.load(stats_dir)
    model = SimVP(INPUT_LEN * C, HORIZON * C, hidden_width=HIDDEN_WIDTH, translator_depth=6).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def forward(x):  # x: (INPUT_LEN, C, PAD_H, PAD_W) normalized -> (HORIZON, C, ORIG_H, ORIG_W) normalized
        t = torch.from_numpy(np.ascontiguousarray(x)).reshape(1, INPUT_LEN * C, PAD_H, PAD_W).to(device)
        with torch.no_grad():
            y = model(t)[..., :ORIG_H, :ORIG_W]
        return y.reshape(HORIZON, C, ORIG_H, ORIG_W).cpu().numpy()

    info = {"checkpoint_epoch": ckpt.get("epoch"), "checkpoint_val_loss": ckpt.get("val_loss")}
    return Forecaster(stats, forward, info)
