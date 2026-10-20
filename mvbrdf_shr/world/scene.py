"""World-space 3D Gaussian scene with physically constrained BRDF materials."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized (w,x,y,z) quaternions to rotation matrices."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


@dataclass(slots=True)
class GaussianMaterials:
    albedo: torch.Tensor
    diffuse_weight: torch.Tensor
    specular: torch.Tensor
    roughness: torch.Tensor
    metalness: torch.Tensor
    absorption: torch.Tensor


class GaussianBRDFField(nn.Module):
    """One persistent material/geometry record per 3D surface Gaussian.

    Energy is enforced by construction: diffuse_weight + specular +
    absorption == 1 through a softmax budget.
    """

    def __init__(
        self,
        means: torch.Tensor,
        colors: torch.Tensor | None = None,
        initial_scale: float = 0.02,
    ):
        super().__init__()
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError("means must have shape (N,3)")
        n = means.shape[0]
        device, dtype = means.device, means.dtype
        colors = (
            colors
            if colors is not None
            else torch.full((n, 3), 0.5, device=device, dtype=dtype)
        )
        colors = colors.to(device=device, dtype=dtype).clamp(1e-4, 1 - 1e-4)

        self.means = nn.Parameter(means.clone())
        self._max_scale = max(1.0, float(initial_scale) * 15.0)
        self.log_scales = nn.Parameter(
            torch.full((n, 3), float(initial_scale), device=device, dtype=dtype).log()
        )
        q = torch.zeros(n, 4, device=device, dtype=dtype)
        q[:, 0] = 1
        self.quaternions = nn.Parameter(q)
        self.opacity_logits = nn.Parameter(
            torch.full((n, 1), 0.8, device=device, dtype=dtype).logit()
        )
        centered = means - means.mean(dim=0, keepdim=True)
        fallback = torch.zeros_like(centered)
        fallback[:, 2] = 1
        radial = torch.where(
            centered.norm(dim=-1, keepdim=True) > 1e-6, centered, fallback
        )
        self.normal_raw = nn.Parameter(F.normalize(radial, dim=-1))
        self.base_color_logits = nn.Parameter(inverse_sigmoid(colors))

        # Initial energy: 75% diffuse, 20% specular, 5% absorbed.
        budget = torch.tensor([0.75, 0.20, 0.05], device=device, dtype=dtype)
        self.material_budget_logits = nn.Parameter(
            budget.log().view(1, 3).expand(n, -1).clone()
        )
        self.roughness_logits = nn.Parameter(
            torch.full((n, 1), -0.85, device=device, dtype=dtype)
        )
        self.metalness_logits = nn.Parameter(
            torch.full((n, 1), -3.0, device=device, dtype=dtype)
        )
        # View-consistent projected texture, initialized from point colors.
        self.texture_logits = nn.Parameter(inverse_sigmoid(colors).clone())

    @property
    def n_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp().clamp(1e-4, getattr(self, "_max_scale", 1.0))

    @property
    def rotations(self) -> torch.Tensor:
        return quaternion_to_matrix(self.quaternions)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def normals(self) -> torch.Tensor:
        return F.normalize(self.normal_raw, dim=-1, eps=1e-6)

    @property
    def texture(self) -> torch.Tensor:
        return torch.sigmoid(self.texture_logits)

    def materials(self) -> GaussianMaterials:
        budget = torch.softmax(self.material_budget_logits, dim=-1)
        diffuse_weight = budget[:, 0:1]
        specular = budget[:, 1:2]
        absorption = budget[:, 2:3]
        base_color = torch.sigmoid(self.base_color_logits)
        albedo = base_color * diffuse_weight
        roughness = 0.04 + 0.96 * torch.sigmoid(self.roughness_logits)
        metalness = torch.sigmoid(self.metalness_logits)
        return GaussianMaterials(
            albedo=albedo,
            diffuse_weight=diffuse_weight,
            specular=specular,
            roughness=roughness,
            metalness=metalness,
            absorption=absorption,
        )

    def covariance_world(self) -> torch.Tensor:
        rotation = self.rotations
        diag = torch.diag_embed(self.scales.square())
        return rotation @ diag @ rotation.transpose(-1, -2)

    def energy_sum(self) -> torch.Tensor:
        m = self.materials()
        return m.diffuse_weight + m.specular + m.absorption

    @classmethod
    def from_point_cloud(
        cls,
        points: torch.Tensor,
        colors: torch.Tensor | None = None,
        initial_scale: float | None = None,
        colors_are_albedo: bool = True,
    ) -> "GaussianBRDFField":
        if initial_scale is None:
            if len(points) > 1:
                sample = points[: min(len(points), 1024)]
                dist = torch.cdist(sample, sample)
                dist.fill_diagonal_(float("inf"))
                initial_scale = float(dist.min(dim=1).values.median().clamp(min=1e-3))
            else:
                initial_scale = 0.02
        scene = cls(points, colors, initial_scale)
        if colors is not None and colors_are_albedo:
            # The constructor accepts unconstrained base color, whereas point
            # clouds conventionally store the final diffuse albedo. Undo the
            # initial 75% diffuse energy budget so the rendered albedo starts
            # at the supplied point colors rather than 25% too dark.
            with torch.no_grad():
                base_color = (colors.to(points) / 0.75).clamp(1e-4, 1 - 1e-4)
                scene.base_color_logits.copy_(inverse_sigmoid(base_color))
        return scene

    def initialize_from_npz(
        self, path: str, mode: str = "geometry"
    ) -> None:
        """Initialize calibrated synthetic priors.

        ``geometry`` imports only scale, rotation and normal while retaining
        generic learnable materials. ``geometry_albedo`` additionally imports
        calibrated diffuse albedo but leaves specular BRDF unknown. ``oracle``
        imports all BRDF parameters and is intended solely as a renderer
        sanity-check upper bound, never as a fair inverse-rendering result.
        """
        if mode not in {"geometry", "geometry_albedo", "oracle"}:
            raise ValueError(
                "mode must be 'geometry', 'geometry_albedo', or 'oracle'"
            )
        import numpy as np

        arrays = np.load(path)

        def tensor(name: str) -> torch.Tensor:
            return torch.from_numpy(arrays[name]).to(self.means)

        if len(arrays["position"]) != self.n_gaussians:
            raise ValueError("Gaussian prior count does not match scene")
        with torch.no_grad():
            self.means.copy_(tensor("position"))
            self.log_scales.copy_(tensor("scale").clamp(min=1e-4).log())
            self.quaternions.copy_(F.normalize(tensor("quaternion"), dim=-1))
            self.normal_raw.copy_(F.normalize(tensor("normal"), dim=-1))
            if "opacity" in arrays:
                self.opacity_logits.copy_(inverse_sigmoid(tensor("opacity")))
            if mode == "geometry_albedo":
                diffuse = self.materials().diffuse_weight
                base_color = (tensor("albedo") / diffuse).clamp(1e-4, 1 - 1e-4)
                self.base_color_logits.copy_(inverse_sigmoid(base_color))
            if mode == "oracle":
                specular = tensor("specular").clamp(1e-4, 1 - 1e-4)
                absorption = tensor("absorption").clamp(1e-4, 1 - 1e-4)
                diffuse = (1 - specular - absorption).clamp(min=1e-4)
                budget = torch.cat([diffuse, specular, absorption], dim=-1)
                budget = budget / budget.sum(dim=-1, keepdim=True)
                self.material_budget_logits.copy_(budget.log())
                albedo = tensor("albedo")
                base_color = (albedo / budget[:, :1]).clamp(1e-4, 1 - 1e-4)
                self.base_color_logits.copy_(inverse_sigmoid(base_color))
                roughness = ((tensor("roughness") - 0.04) / 0.96).clamp(
                    1e-4, 1 - 1e-4
                )
                self.roughness_logits.copy_(inverse_sigmoid(roughness))
                self.metalness_logits.copy_(
                    inverse_sigmoid(tensor("metalness"))
                )

    def geometry_parameters(self) -> list[nn.Parameter]:
        return [
            self.means,
            self.log_scales,
            self.quaternions,
            self.opacity_logits,
            self.normal_raw,
        ]

    def material_parameters(self) -> list[nn.Parameter]:
        return [
            self.base_color_logits,
            self.material_budget_logits,
            self.roughness_logits,
            self.metalness_logits,
            self.texture_logits,
        ]

