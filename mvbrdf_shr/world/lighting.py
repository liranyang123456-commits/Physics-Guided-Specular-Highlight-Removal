"""Per-frame SH/directional lighting and a NeILF-style incident light field."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class WorldLight:
    env_sh: torch.Tensor      # (9,3)
    direction: torch.Tensor   # (3,), surface-to-light
    intensity: torch.Tensor   # (1,)
    color: torch.Tensor       # (3,)
    position: torch.Tensor    # (3,)
    point_weight: torch.Tensor  # (1,), 0=directional, 1=point


class PerFrameLighting(nn.Module):
    """First-level lighting: SH environment plus one dominant light per frame."""

    def __init__(self, n_frames: int, shared: bool = False):
        super().__init__()
        self.shared = shared
        sh = torch.zeros(n_frames, 9, 3)
        sh[:, 0] = 1.0
        self.env_sh = nn.Parameter(sh)
        direction = torch.tensor([0.3, -0.4, 0.85]).view(1, 3).repeat(n_frames, 1)
        self.direction_raw = nn.Parameter(direction)
        self.log_intensity = nn.Parameter(torch.zeros(n_frames, 1))
        self.color_logits = nn.Parameter(torch.full((n_frames, 3), 2.2))
        self.position = nn.Parameter(
            torch.tensor([0.0, -2.0, 2.0]).view(1, 3).repeat(n_frames, 1)
        )
        self.point_light_logits = nn.Parameter(torch.full((n_frames, 1), -8.0))

    def forward(self, frame_id: int | torch.Tensor) -> WorldLight:
        if self.shared:
            frame_id = 0
        return WorldLight(
            env_sh=self.env_sh[frame_id],
            direction=F.normalize(self.direction_raw[frame_id], dim=-1),
            intensity=self.log_intensity[frame_id].exp(),
            color=torch.sigmoid(self.color_logits[frame_id]),
            position=self.position[frame_id],
            point_weight=torch.sigmoid(self.point_light_logits[frame_id]),
        )


def positional_encoding(x: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    values = [x]
    for k in range(n_freqs):
        f = (2.0**k) * torch.pi
        values.extend([torch.sin(f * x), torch.cos(f * x)])
    return torch.cat(values, dim=-1)


class NeuralIncidentLightField(nn.Module):
    """NeILF-style Li(x, wi): 3D position + 2D direction manifold -> RGB.

    The implementation accepts a 3D unit direction (rather than explicitly
    storing two angles); the direction still has two degrees of freedom.
    """

    def __init__(self, hidden: int = 128, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        encoded_dim = 6 * (1 + 2 * n_freqs)
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(
        self, positions: torch.Tensor, incident_directions: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast positions/directions and return non-negative incident RGB."""
        x, d = torch.broadcast_tensors(positions, incident_directions)
        features = torch.cat(
            [
                positional_encoding(x, self.n_freqs),
                positional_encoding(F.normalize(d, dim=-1), self.n_freqs),
            ],
            dim=-1,
        )
        return F.softplus(self.net(features))


def fibonacci_sphere(
    n_samples: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Deterministic, approximately uniform directions on S²."""
    i = torch.arange(n_samples, device=device, dtype=dtype) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    radius = torch.sqrt((1.0 - z * z).clamp(min=0))
    golden = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, device=device, dtype=dtype)))
    theta = golden * i
    return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), z], dim=-1)

