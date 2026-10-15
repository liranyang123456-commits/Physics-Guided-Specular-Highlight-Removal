"""End-to-end world-space 3D Gaussian BRDF highlight-removal pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn

from .camera import PerspectiveCamera
from .lighting import NeuralIncidentLightField, PerFrameLighting
from .refiner import PhysicsGuidedWorldRefiner
from .renderer import GaussianBRDFRenderer
from .scene import GaussianBRDFField


class WorldGaussianBRDFPipeline(nn.Module):
    """Persistent 3D scene -> physical decomposition -> optional refinement."""

    def __init__(
        self,
        scene: GaussianBRDFField,
        n_frames: int,
        lighting_mode: str = "sh",
        use_refiner: bool = True,
        refiner_base: int = 96,
        chunk_size: int = 64,
        raster_backend: str = "auto",
        shared_lighting: bool = False,
        background_color: tuple[float, float, float] | list[float] | None = None,
    ):
        super().__init__()
        if lighting_mode not in {"sh", "neilf"}:
            raise ValueError("lighting_mode must be 'sh' or 'neilf'")
        self.scene = scene
        self.lighting_mode = lighting_mode
        self.register_buffer(
            "background_color",
            None
            if background_color is None
            else torch.tensor(background_color, dtype=torch.float32).view(3, 1, 1),
            persistent=False,
        )
        self.frame_lighting = PerFrameLighting(n_frames, shared=shared_lighting)
        self.neural_light = (
            NeuralIncidentLightField() if lighting_mode == "neilf" else None
        )
        self.renderer = GaussianBRDFRenderer(
            chunk_size=chunk_size, backend=raster_backend
        )
        self.refiner = (
            PhysicsGuidedWorldRefiner(base=refiner_base) if use_refiner else None
        )

    def forward(
        self,
        camera: PerspectiveCamera,
        frame_id: int,
        image: torch.Tensor | None = None,
        refine: bool = True,
    ) -> dict:
        if self.lighting_mode == "neilf":
            render = self.renderer(
                self.scene, camera, neural_light=self.neural_light
            )
        else:
            render = self.renderer(
                self.scene, camera, light=self.frame_lighting(frame_id)
            )
        if self.background_color is not None:
            background = self.background_color * (1 - render.alpha)
            render.full = render.full + background
            render.diffuse = render.diffuse + background
        pred = render.diffuse
        if refine and self.refiner is not None and image is not None:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            pred = self.refiner(
                image,
                render.diffuse.unsqueeze(0),
                render.mask.unsqueeze(0),
                render.normal.unsqueeze(0),
                render.depth.unsqueeze(0),
                render.projected_texture.unsqueeze(0),
                render.albedo.unsqueeze(0),
                render.alpha.unsqueeze(0),
            )[0]
        return {
            "render": render,
            "pred": pred,
            "image": image[0] if image is not None and image.ndim == 4 else image,
            "materials": self.scene.materials(),
        }

    def load_shiq_refiner(self, checkpoint: str) -> int:
        if self.refiner is None:
            raise RuntimeError("pipeline has no refiner")
        return self.refiner.load_shiq_restorer(checkpoint)

