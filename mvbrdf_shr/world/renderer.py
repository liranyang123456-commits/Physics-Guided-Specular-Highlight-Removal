"""Differentiable anisotropic 3D Gaussian splatting with BRDF separation."""
from __future__ import annotations

from dataclasses import dataclass
import glob
import math
import os
import shutil
import subprocess
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension

if os.name == "nt":
    # gsplat builds its CUDA extension at first use; a localized (non-UTF-8)
    # MSVC banner otherwise raises UnicodeDecodeError and, with backend="auto",
    # silently falls back to the PyTorch compositor.
    torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

from ..models.envlight import sh_to_irradiance
from .camera import PerspectiveCamera
from .lighting import NeuralIncidentLightField, WorldLight, fibonacci_sphere
from .scene import GaussianBRDFField, GaussianMaterials

_PI = 3.141592653589793
_EPS = 1e-6


@dataclass(slots=True)
class WorldRenderOutputs:
    full: torch.Tensor
    diffuse: torch.Tensor
    specular: torch.Tensor
    mask: torch.Tensor
    normal: torch.Tensor
    depth: torch.Tensor
    albedo: torch.Tensor
    projected_texture: torch.Tensor
    alpha: torch.Tensor


def _schlick_fresnel(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return f0 + (1 - f0) * (1 - cos_theta).clamp(0, 1).pow(5)


def _ggx_d(n_dot_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a2 = alpha.square()
    denom = n_dot_h.square() * (a2 - 1) + 1
    return a2 / (_PI * denom.square() + _EPS)


def _smith_g1(cosine: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    k = (alpha + 1).square() / 8
    return cosine / (cosine * (1 - k) + k + _EPS)


class GaussianBRDFRenderer(nn.Module):
    """Reference PyTorch renderer.

    It uses exact projected anisotropic covariance and front-to-back alpha
    compositing.  `chunk_size` bounds memory while preserving depth order.
    For large production scenes this interface can later be backed by gsplat
    without changing the scene or training API.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        min_covariance: float = 0.3,
        mask_threshold: float = 0.08,
        neural_light_samples: int = 32,
        backend: str = "auto",
    ):
        super().__init__()
        if backend not in {"auto", "torch", "gsplat", "dgr"}:
            raise ValueError("backend must be auto, torch, gsplat, or dgr")
        self.chunk_size = chunk_size
        self.min_covariance = min_covariance
        self.mask_threshold = nn.Parameter(torch.tensor(float(mask_threshold)))
        self.neural_light_samples = neural_light_samples
        self.backend = backend
        self._gsplat_failed = False

    @staticmethod
    def gsplat_available() -> bool:
        try:
            from gsplat import rasterization  # noqa: F401

            return True
        except (ImportError, OSError):
            return False

    def active_backend(self, device: torch.device) -> str:
        if self.backend == "torch":
            return "torch"
        if self.backend == "dgr":
            if device.type != "cuda":
                raise RuntimeError("dgr backend requires CUDA tensors")
            return "dgr"
        if self.backend == "gsplat":
            if device.type != "cuda":
                raise RuntimeError("gsplat backend requires CUDA tensors")
            if not self.gsplat_available():
                raise RuntimeError("gsplat backend requested but gsplat is unavailable")
            return "gsplat"
        # auto: prefer gsplat, then dgr (works on sm_120/Blackwell), then torch.
        if device.type == "cuda":
            if not self._gsplat_failed and self.gsplat_available():
                return "gsplat"
            if self.dgr_available():
                return "dgr"
        return "torch"

    @staticmethod
    def dgr_available() -> bool:
        try:
            import diff_gaussian_rasterization  # noqa: F401

            return True
        except Exception:
            return False

    def _shade_sh(
        self,
        scene: GaussianBRDFField,
        materials: GaussianMaterials,
        camera: PerspectiveCamera,
        light: WorldLight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = scene.normals
        view = F.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
        directional = F.normalize(light.direction, dim=-1).view(1, 3).expand_as(n)
        point_vector = light.position.view(1, 3) - scene.means
        point_distance = point_vector.norm(dim=-1, keepdim=True).clamp(min=0.1)
        point_direction = point_vector / point_distance
        point_weight = light.point_weight.view(1, 1)
        direction = F.normalize(
            directional * (1 - point_weight) + point_direction * point_weight,
            dim=-1,
        )
        n_dot_l = (n * direction).sum(-1, keepdim=True).clamp(min=0)
        n_dot_v = (n * view).sum(-1, keepdim=True).clamp(min=1e-4)

        # SH diffuse irradiance plus the dominant direct light.
        irradiance = sh_to_irradiance(
            n.T.unsqueeze(0).unsqueeze(-1), light.env_sh.unsqueeze(0)
        )[0, :, :, 0].T
        attenuation = (1 - point_weight) + point_weight / point_distance.square()
        direct = (
            light.color.view(1, 3)
            * light.intensity.view(1, 1)
            * attenuation
            * n_dot_l
        )
        diffuse = materials.albedo / _PI * (irradiance + direct)

        half = F.normalize(direction + view, dim=-1)
        n_dot_h = (n * half).sum(-1, keepdim=True).clamp(min=0)
        h_dot_v = (half * view).sum(-1, keepdim=True).clamp(min=0)
        alpha = materials.roughness.square()
        d_term = _ggx_d(n_dot_h, alpha)
        g_term = _smith_g1(n_dot_l, alpha) * _smith_g1(n_dot_v, alpha)
        f0 = 0.04 * (1 - materials.metalness) + materials.albedo * materials.metalness
        fresnel = _schlick_fresnel(h_dot_v, f0)
        spec_brdf = (
            materials.specular * d_term * g_term * fresnel
            / (4 * n_dot_l * n_dot_v + _EPS)
        )
        specular = spec_brdf * direct
        return diffuse, specular

    def _shade_neural(
        self,
        scene: GaussianBRDFField,
        materials: GaussianMaterials,
        camera: PerspectiveCamera,
        field: NeuralIncidentLightField,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Monte-Carlo rendering-equation integration using Li(x, wi)."""
        g = scene.n_gaussians
        dirs = fibonacci_sphere(
            self.neural_light_samples, scene.means.device, scene.means.dtype
        )
        wi = dirs.view(1, -1, 3).expand(g, -1, -1)
        x = scene.means[:, None, :].expand_as(wi)
        li = field(x, wi)
        n = scene.normals[:, None, :]
        view = F.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
        view = view[:, None, :]
        n_dot_l = (n * wi).sum(-1, keepdim=True).clamp(min=0)
        n_dot_v = (n * view).sum(-1, keepdim=True).clamp(min=1e-4)

        diffuse_brdf = materials.albedo[:, None, :] / _PI
        half = F.normalize(wi + view, dim=-1)
        n_dot_h = (n * half).sum(-1, keepdim=True).clamp(min=0)
        h_dot_v = (half * view).sum(-1, keepdim=True).clamp(min=0)
        alpha = materials.roughness[:, None, :].square()
        d_term = _ggx_d(n_dot_h, alpha)
        g_term = _smith_g1(n_dot_l, alpha) * _smith_g1(n_dot_v, alpha)
        f0 = (
            0.04 * (1 - materials.metalness) + materials.albedo * materials.metalness
        )[:, None, :]
        fresnel = _schlick_fresnel(h_dot_v, f0)
        spec_brdf = (
            materials.specular[:, None, :]
            * d_term
            * g_term
            * fresnel
            / (4 * n_dot_l * n_dot_v + _EPS)
        )
        solid_angle = 4 * _PI / self.neural_light_samples
        diffuse = (li * diffuse_brdf * n_dot_l).sum(1) * solid_angle
        specular = (li * spec_brdf * n_dot_l).sum(1) * solid_angle
        return diffuse, specular

    def _project_covariance(
        self, scene: GaussianBRDFField, camera: PerspectiveCamera
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points_camera = camera.world_to_camera(scene.means)
        uv, depth, valid = camera.project(scene.means)
        rotation_cw = camera.w2c[:3, :3]
        cov_world = scene.covariance_world()
        cov_camera = rotation_cw @ cov_world @ rotation_cw.T
        jacobian = camera.projection_jacobian(points_camera)
        cov_2d = jacobian @ cov_camera @ jacobian.transpose(-1, -2)
        eye = torch.eye(2, device=cov_2d.device, dtype=cov_2d.dtype)
        cov_2d = cov_2d + self.min_covariance * eye
        return uv, depth, valid, cov_2d

    def _splat_torch(
        self,
        scene: GaussianBRDFField,
        camera: PerspectiveCamera,
        attributes: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        uv, depth, valid, covariance = self._project_covariance(scene, camera)
        ids = torch.where(valid)[0]
        if ids.numel() == 0:
            empty = {
                name: torch.zeros(
                    value.shape[-1],
                    camera.height,
                    camera.width,
                    device=value.device,
                    dtype=value.dtype,
                )
                for name, value in attributes.items()
            }
            alpha = torch.zeros(
                1,
                camera.height,
                camera.width,
                device=scene.means.device,
                dtype=scene.means.dtype,
            )
            return empty, alpha

        ids = ids[torch.argsort(depth[ids])]
        ys, xs = torch.meshgrid(
            torch.arange(
                camera.height, device=scene.means.device, dtype=scene.means.dtype
            ),
            torch.arange(
                camera.width, device=scene.means.device, dtype=scene.means.dtype
            ),
            indexing="ij",
        )
        pixels = torch.stack([xs, ys], dim=-1)
        transmittance = torch.ones_like(xs)
        rendered = {
            name: torch.zeros(
                value.shape[-1],
                camera.height,
                camera.width,
                device=value.device,
                dtype=value.dtype,
            )
            for name, value in attributes.items()
        }

        for start in range(0, ids.numel(), self.chunk_size):
            chunk = ids[start : start + self.chunk_size]
            delta = pixels.unsqueeze(0) - uv[chunk, None, None, :]
            inv_cov = torch.linalg.inv(covariance[chunk])
            mahalanobis = torch.einsum(
                "ghwi,gij,ghwj->ghw", delta, inv_cov, delta
            )
            alpha = (
                scene.opacity[chunk, 0, None, None]
                * torch.exp(-0.5 * mahalanobis.clamp(min=0))
            ).clamp(0, 0.99)
            prefix = torch.cumprod(
                torch.cat([torch.ones_like(alpha[:1]), 1 - alpha + 1e-7], dim=0),
                dim=0,
            )[:-1]
            weights = transmittance.unsqueeze(0) * prefix * alpha
            for name, value in attributes.items():
                rendered[name] = rendered[name] + torch.einsum(
                    "ghw,gc->chw", weights, value[chunk]
                )
            transmittance = transmittance * torch.prod(1 - alpha + 1e-7, dim=0)

        return rendered, (1 - transmittance).unsqueeze(0)

    def _splat_gsplat(
        self,
        scene: GaussianBRDFField,
        camera: PerspectiveCamera,
        attributes: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """CUDA tile rasterization of all G-buffers in one gsplat call."""
        # User-site installs on Windows do not always append Scripts to PATH,
        # although the ninja wheel installed ninja.exe there.
        if shutil.which("ninja") is None:
            try:
                import ninja

                os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ["PATH"]
            except ImportError:
                pass
        if os.name == "nt" and shutil.which("cl") is None:
            candidates = (
                glob.glob(
                    r"C:\Program Files\Microsoft Visual Studio\2022"
                    r"\*\VC\Auxiliary\Build\vcvars64.bat"
                )
                + glob.glob(
                    r"C:\Program Files (x86)\Microsoft Visual Studio\2019"
                    r"\*\VC\Auxiliary\Build\vcvars64.bat"
                )
            )
            if candidates:
                environment = subprocess.check_output(
                    f'call "{candidates[0]}" >nul && set',
                    shell=True,
                    text=True,
                    encoding="mbcs",
                )
                for line in environment.splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value
        if os.name == "nt":
            compatibility_header = os.path.join(
                os.path.dirname(__file__), "gsplat_windows_compat.h"
            )
            nvcc_flags = os.environ.get("NVCC_PREPEND_FLAGS", "")
            preinclude = f'--pre-include "{compatibility_header}"'
            if preinclude not in nvcc_flags:
                os.environ["NVCC_PREPEND_FLAGS"] = (
                    f"{preinclude} {nvcc_flags}".strip()
                )
            # gsplat 1.5 emits GCC host flags even on Windows. Translate them
            # before its lazy backend imports PyTorch's private JIT helper.
            import torch.utils.cpp_extension as cpp_extension

            if not getattr(cpp_extension._jit_compile, "_mvbrdf_msvc", False):
                original_jit_compile = cpp_extension._jit_compile

                def msvc_jit_compile(*args, **kwargs):
                    positional = list(args)
                    if len(positional) > 2 and positional[2] is not None:
                        flags = []
                        for flag in positional[2]:
                            if flag == "-O3":
                                flags.append("/O2")
                            elif flag == "-O0":
                                flags.append("/Od")
                            elif not flag.startswith("-W"):
                                flags.append(flag)
                        positional[2] = flags
                    return original_jit_compile(*positional, **kwargs)

                msvc_jit_compile._mvbrdf_msvc = True
                cpp_extension._jit_compile = msvc_jit_compile
        from gsplat import rasterization

        names = [name for name in attributes if name != "depth"]
        widths = [attributes[name].shape[-1] for name in names]
        features = torch.cat([attributes[name] for name in names], dim=-1)
        rendered_features, rendered_alpha, _ = rasterization(
            means=scene.means,
            quats=scene.quaternions,
            scales=scene.scales,
            opacities=scene.opacity[:, 0],
            colors=features,
            viewmats=camera.w2c.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.width,
            height=camera.height,
            eps2d=self.min_covariance,
            packed=True,
            render_mode="RGB+ED",
            channel_chunk=32,
        )
        # gsplat output is (camera, H, W, channels); ED is the last channel.
        channels = rendered_features[0, ..., :-1]
        depth = rendered_features[0, ..., -1:].permute(2, 0, 1)
        rendered: dict[str, torch.Tensor] = {"depth": depth}
        offset = 0
        for name, width in zip(names, widths):
            rendered[name] = channels[..., offset : offset + width].permute(2, 0, 1)
            offset += width
        alpha = rendered_alpha[0].permute(2, 0, 1)
        return rendered, alpha

    def _splat_dgr(
        self,
        scene: GaussianBRDFField,
        camera: PerspectiveCamera,
        attributes: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """diff-gaussian-rasterization backend (sm_90 build runs on sm_120).

        Used when gsplat is unavailable (e.g. RTX 5090 / Blackwell sm_120 where
        gsplat's JIT fails). DGR only renders 3-channel RGB + alpha + depth per
        call, so we invoke it once per 3-channel attribute and reuse the alpha.
        Slower than gsplat's single multi-channel call but ~30x faster than the
        pure-PyTorch fallback, and works on sm_120.
        """
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        device = scene.means.device
        dtype = torch.float32
        means3d = scene.means.to(device=device, dtype=dtype)
        # DGR expects scales as ACTUAL values (exp already applied), matching
        # 3DGS GaussianModel.get_scaling = torch.exp(self._scaling). scene.scales
        # is already the activated (exp+clamp) value, so pass it directly.
        scales = scene.scales.to(device=device, dtype=dtype)
        rotations = torch.nn.functional.normalize(
            scene.quaternions.to(device=device, dtype=dtype), dim=-1
        )
        opacities = scene.opacity[:, 0].to(device=device, dtype=dtype)

        # Build view/projection matrices in the convention diff-gaussian-rasterization
        # expects (3DGS standard): viewmatrix = w2c (row-major), projmatrix from the
        # FOV-based getProjectionMatrix (NOT the intrinsics K), then full = P @ W.
        # DGR stores matrices transposed (column-major), so we pass .T.
        w2c = camera.w2c.to(device=device, dtype=dtype)
        fx, fy = camera.K[0, 0].item(), camera.K[1, 1].item()
        fovx = 2.0 * math.atan(float(camera.width) / (2.0 * fx))
        fovy = 2.0 * math.atan(float(camera.height) / (2.0 * fy))
        # Choose a near/far that brackets the scene (world objects ~unit scale).
        znear, zfar = 0.01, 100.0
        tan_half_fovx = math.tan(fovx / 2.0)
        tan_half_fovy = math.tan(fovy / 2.0)
        top = tan_half_fovy * znear
        right = tan_half_fovx * znear
        P = torch.zeros(4, 4, device=device, dtype=dtype)
        P[0, 0] = 2.0 * znear / (2.0 * right)
        P[1, 1] = 2.0 * znear / (2.0 * top)
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0
        # viewmatrix (w2c) and full projection = P @ w2c, both transposed for DGR.
        viewmat = w2c.T
        projmat = (P @ w2c).T

        background = torch.zeros(3, device=device, dtype=dtype)

        def rasterize_attribute(colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Rasterize one (N,3) attribute; returns (out (3,H,W), alpha (1,H,W))."""
            settings = GaussianRasterizationSettings(
                image_height=camera.height,
                image_width=camera.width,
                tanfovx=tan_half_fovx,
                tanfovy=tan_half_fovy,
                bg=background,
                scale_modifier=1.0,
                viewmatrix=viewmat,
                projmatrix=projmat,
                sh_degree=0,
                campos=camera.c2w[:3, 3].to(device=device, dtype=dtype),
                prefiltered=False,
                debug=False,
                antialiasing=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=settings)
            out, alpha = rasterizer(
                means3D=means3d,
                means2D=torch.zeros_like(means3d[:, :2]),
                shs=None,
                colors_precomp=colors.to(device=device, dtype=dtype),
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
            return out, alpha[0:1]

        rendered: dict[str, torch.Tensor] = {}
        first_alpha = None
        for name, value in attributes.items():
            if name == "depth":
                continue  # depth handled separately below if needed
            colors = value.to(device=device, dtype=dtype)
            if colors.shape[-1] != 3:
                # Pad/truncate non-3-channel attributes to 3 for rasterization.
                if colors.shape[-1] < 3:
                    pad = torch.zeros(colors.shape[0], 3 - colors.shape[-1], device=device, dtype=dtype)
                    colors = torch.cat([colors, pad], dim=-1)
                else:
                    colors = colors[..., :3]
            out, alpha = rasterize_attribute(colors)
            # Trim back to original channel count if we padded.
            ch = value.shape[-1]
            rendered[name] = out[:ch]
            if first_alpha is None:
                first_alpha = alpha
        if first_alpha is None:
            first_alpha = torch.zeros(1, camera.height, camera.width, device=device, dtype=dtype)
        # depth: approximate via DGR's means2D z (optional; torch fallback used it).
        rendered["depth"] = torch.zeros(1, camera.height, camera.width, device=device, dtype=dtype)
        return rendered, first_alpha

    def forward(
        self,
        scene: GaussianBRDFField,
        camera: PerspectiveCamera,
        light: WorldLight | None = None,
        neural_light: NeuralIncidentLightField | None = None,
    ) -> WorldRenderOutputs:
        if neural_light is None and light is None:
            raise ValueError("either light or neural_light is required")
        materials = scene.materials()
        if neural_light is not None:
            diffuse_g, specular_g = self._shade_neural(
                scene, materials, camera, neural_light
            )
        else:
            assert light is not None
            diffuse_g, specular_g = self._shade_sh(
                scene, materials, camera, light
            )
        full_g = diffuse_g + specular_g
        _, depth_g, _, _ = self._project_covariance(scene, camera)
        attrs = {
            "full": full_g,
            "diffuse": diffuse_g,
            "specular": specular_g,
            "normal": scene.normals,
            "depth": depth_g[:, None],
            "albedo": materials.albedo,
            "texture": scene.texture,
        }
        backend = self.active_backend(scene.means.device)
        if backend == "gsplat":
            try:
                rendered, alpha = self._splat_gsplat(scene, camera, attrs)
            except Exception as error:
                if self.backend == "gsplat":
                    raise
                self._gsplat_failed = True
                # Fall back to dgr (sm_120) or torch.
                if self.dgr_available():
                    warnings.warn(
                        f"gsplat unavailable at runtime ({error}); falling back to "
                        "diff-gaussian-rasterization.",
                        RuntimeWarning,
                    )
                    rendered, alpha = self._splat_dgr(scene, camera, attrs)
                else:
                    warnings.warn(
                        f"gsplat unavailable at runtime ({error}); falling back to "
                        "the differentiable PyTorch rasterizer.",
                        RuntimeWarning,
                    )
                    rendered, alpha = self._splat_torch(scene, camera, attrs)
        elif backend == "dgr":
            rendered, alpha = self._splat_dgr(scene, camera, attrs)
        else:
            rendered, alpha = self._splat_torch(scene, camera, attrs)
        denom = alpha.clamp(min=1e-6)
        normal = F.normalize(rendered["normal"] / denom, dim=0, eps=1e-6)
        depth = rendered["depth"] / denom
        threshold = self.mask_threshold.abs().clamp(min=1e-4)
        mask = (rendered["specular"].mean(0, keepdim=True) / threshold).clamp(0, 1)
        return WorldRenderOutputs(
            full=rendered["full"].clamp(0, 1),
            diffuse=rendered["diffuse"].clamp(0, 1),
            specular=rendered["specular"].clamp(0, 1),
            mask=mask,
            normal=normal,
            depth=depth,
            albedo=rendered["albedo"].clamp(0, 1),
            projected_texture=rendered["texture"].clamp(0, 1),
            alpha=alpha,
        )

