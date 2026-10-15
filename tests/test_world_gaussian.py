from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.data import MultiViewScene
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR
from mvbrdf_shr.world.lighting import NeuralIncidentLightField, PerFrameLighting
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer, WorldRenderOutputs
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.train import split_frame_indices


def camera(size: int = 32) -> PerspectiveCamera:
    K = torch.tensor(
        [[30.0, 0, size / 2], [0, 30.0, size / 2], [0, 0, 1.0]]
    )
    return PerspectiveCamera(K, torch.eye(4), size, size)


def scene() -> GaussianBRDFField:
    points = torch.tensor(
        [
            [-0.25, -0.2, 2.0],
            [0.25, -0.2, 2.0],
            [-0.2, 0.25, 2.1],
            [0.2, 0.25, 2.1],
        ]
    )
    colors = torch.tensor(
        [[0.8, 0.2, 0.1], [0.2, 0.8, 0.1], [0.1, 0.2, 0.8], [0.7, 0.7, 0.2]]
    )
    return GaussianBRDFField(points, colors, initial_scale=0.12)


def test_camera_projection_and_rays():
    cam = camera()
    uv, depth, valid = cam.project(torch.tensor([[0.0, 0.0, 2.0]]))
    assert torch.allclose(uv[0], torch.tensor([16.0, 16.0]))
    assert depth.item() == 2.0
    assert valid.item()
    origins, directions = cam.pixel_rays()
    assert origins.shape == directions.shape == (32, 32, 3)
    assert torch.allclose(directions.norm(dim=-1), torch.ones(32, 32), atol=1e-5)


def test_material_energy_is_conserved():
    field = scene()
    assert torch.allclose(field.energy_sum(), torch.ones(4, 1), atol=1e-6)
    materials = field.materials()
    assert materials.albedo.shape == (4, 3)
    assert materials.roughness.min() >= 0.04


def test_point_cloud_colors_initialize_rendered_albedo():
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    colors = torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.2, 0.1]])
    field = GaussianBRDFField.from_point_cloud(points, colors, initial_scale=0.1)
    assert torch.allclose(field.materials().albedo, colors, atol=1e-4)


def test_geometry_npz_initialization_and_frame_split(tmp_path):
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    field = GaussianBRDFField.from_point_cloud(points, initial_scale=0.1)
    scale = np.full((2, 3), 0.03, dtype=np.float32)
    quaternion = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    normal = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
    prior = tmp_path / "prior.npz"
    np.savez(
        prior,
        position=points.numpy(),
        scale=scale,
        quaternion=quaternion,
        normal=normal,
    )
    field.initialize_from_npz(str(prior), mode="geometry")
    assert torch.allclose(field.scales, torch.from_numpy(scale))
    assert torch.allclose(field.normals, torch.from_numpy(normal))
    train, holdout = split_frame_indices(
        12, {"holdout_stride": 4, "holdout_offset": 0}
    )
    assert holdout == [0, 4, 8]
    assert set(train).isdisjoint(holdout)
    assert sorted(train + holdout) == list(range(12))


def test_anisotropic_renderer_outputs_and_gradients():
    field = scene()
    cam = camera()
    lighting = PerFrameLighting(1)
    renderer = GaussianBRDFRenderer(chunk_size=2)
    output = renderer(field, cam, light=lighting(0))
    assert output.full.shape == (3, 32, 32)
    assert output.diffuse.shape == output.specular.shape == output.full.shape
    assert output.depth.shape == output.mask.shape == (1, 32, 32)
    assert output.normal.shape == (3, 32, 32)
    loss = output.full.mean() + output.diffuse.mean()
    loss.backward()
    assert field.means.grad is not None
    assert field.base_color_logits.grad is not None


def test_neilf_renderer_and_pipeline():
    field = scene()
    cam = camera(24)
    neilf = NeuralIncidentLightField(hidden=32, n_freqs=2)
    renderer = GaussianBRDFRenderer(chunk_size=4, neural_light_samples=8)
    output = renderer(field, cam, neural_light=neilf)
    assert output.full.shape == (3, 24, 24)
    pipeline = WorldGaussianBRDFPipeline(
        field, n_frames=1, lighting_mode="sh", use_refiner=False, chunk_size=4
    )
    result = pipeline(cam, 0, image=torch.zeros(3, 24, 24), refine=False)
    assert torch.allclose(result["pred"], result["render"].diffuse)


def test_hybrid_falls_back_to_single_and_rejects_invisible_world():
    class Single(torch.nn.Module):
        def forward(self, image, **_):
            mask = torch.full_like(image[:, :1], 0.2)
            return {
                "pred": image * 0.5,
                "spec_pred": image * 0.1,
                "mask_pred": mask,
            }

    hybrid = HybridMVBRDFSHR(single=Single(), gate_hidden=8)
    image = torch.rand(1, 3, 8, 8)
    single = hybrid(image)
    assert single["mode"] == "single"
    assert torch.allclose(single["pred"], image * 0.5)
    zeros3 = torch.zeros(3, 8, 8)
    zeros1 = torch.zeros(1, 8, 8)
    world = WorldRenderOutputs(
        full=zeros3,
        diffuse=zeros3,
        specular=zeros3,
        mask=zeros1,
        normal=zeros3,
        depth=zeros1,
        albedo=zeros3,
        projected_texture=zeros3,
        alpha=zeros1,
    )
    fused = hybrid(image, world)
    assert fused["mode"] == "hybrid"
    assert float(fused["gate"].detach().max()) < 1e-3
    assert torch.allclose(fused["pred"], image * 0.5, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gsplat_matches_reference_and_has_gradients():
    if not GaussianBRDFRenderer.gsplat_available():
        pytest.skip("gsplat unavailable")
    field = scene().cuda()
    cam = camera().to("cuda")
    lighting = PerFrameLighting(1).cuda()
    reference = GaussianBRDFRenderer(chunk_size=2, backend="torch")(
        field, cam, light=lighting(0)
    )
    accelerated = GaussianBRDFRenderer(backend="gsplat")(
        field, cam, light=lighting(0)
    )
    assert (reference.full - accelerated.full).abs().mean() < 0.01
    assert (reference.diffuse - accelerated.diffuse).abs().mean() < 0.01
    accelerated.full.mean().backward()
    assert field.means.grad is not None
    assert torch.isfinite(field.means.grad).all()


def test_blender_transforms_loader(tmp_path):
    images = tmp_path / "images"
    diffuse = tmp_path / "diffuse"
    images.mkdir()
    diffuse.mkdir()
    array = np.full((16, 16, 3), 128, dtype=np.uint8)
    Image.fromarray(array).save(images / "0000.png")
    Image.fromarray(array // 2).save(diffuse / "0000.png")
    meta = {
        "camera_angle_x": 0.7,
        "frames": [
            {
                "file_path": "images/0000.png",
                "diffuse_path": "diffuse/0000.png",
                "transform_matrix": np.eye(4).tolist(),
                "light_direction": [0, 0, 1],
                "light_intensity": 1.0,
            }
        ],
    }
    (tmp_path / "transforms.json").write_text(json.dumps(meta), encoding="utf-8")
    dataset = MultiViewScene.from_blender(tmp_path)
    assert len(dataset) == 1
    assert dataset[0].image.shape == (3, 16, 16)
    assert dataset[0].diffuse_gt is not None
    assert dataset[0].camera.width == 16
    pseudo = torch.full((3, 16, 16), 0.25)
    confidence = torch.ones(1, 16, 16)
    origins, directions = dataset[0].camera.pixel_rays()
    point = origins[8, 8] + 2 * directions[8, 8]
    fused = dataset.fuse_projected_texture(
        point.unsqueeze(0),
        [0],
        source_images={0: pseudo},
        confidence_maps={0: confidence},
    )
    assert torch.allclose(fused, torch.full((1, 3), 0.25), atol=1e-5)


def test_mvsnet_loader_unprojects_depth(tmp_path):
    images = tmp_path / "blended_images"
    cameras = tmp_path / "cams"
    depths = tmp_path / "rendered_depth_maps"
    images.mkdir()
    cameras.mkdir()
    depths.mkdir()
    Image.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8)).save(
        images / "00000000.jpg"
    )
    camera_text = """extrinsic
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

intrinsic
4 0 4
0 4 3
0 0 1

0.1 0.01 10 2.0
"""
    (cameras / "00000000_cam.txt").write_text(camera_text)
    depth = np.full((6, 8), 2.0, dtype="<f4")
    with (depths / "00000000.pfm").open("wb") as file:
        file.write(b"Pf\n8 6\n-1.0\n")
        np.flipud(depth).tofile(file)
    dataset = MultiViewScene.from_mvsnet(
        tmp_path, scale=1.0, max_points=12, max_point_views=1
    )
    assert len(dataset) == 1
    assert dataset.points is not None and len(dataset.points) == 12
    assert torch.allclose(dataset.points[:, 2], torch.full((12,), 2.0))
    assert dataset.point_colors is not None

