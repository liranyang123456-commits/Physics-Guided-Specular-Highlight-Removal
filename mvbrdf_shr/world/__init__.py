"""True world-space multi-view 3D Gaussian BRDF subsystem."""

from .camera import PerspectiveCamera
from .data import MultiViewFrame, MultiViewScene
from .fusion import HybridMVBRDFSHR
from .lighting import NeuralIncidentLightField, PerFrameLighting, WorldLight
from .pipeline import WorldGaussianBRDFPipeline
from .renderer import GaussianBRDFRenderer, WorldRenderOutputs
from .scene import GaussianBRDFField, GaussianMaterials

__all__ = [
    "PerspectiveCamera",
    "MultiViewFrame",
    "MultiViewScene",
    "HybridMVBRDFSHR",
    "NeuralIncidentLightField",
    "PerFrameLighting",
    "WorldLight",
    "WorldGaussianBRDFPipeline",
    "GaussianBRDFRenderer",
    "WorldRenderOutputs",
    "GaussianBRDFField",
    "GaussianMaterials",
]

