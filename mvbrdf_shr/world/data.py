"""COLMAP and Blender multiview scene loaders."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .camera import PerspectiveCamera


@dataclass(slots=True)
class MultiViewFrame:
    image: torch.Tensor
    camera: PerspectiveCamera
    image_path: Path
    diffuse_gt: torch.Tensor | None = None
    light_position: torch.Tensor | None = None
    light_direction: torch.Tensor | None = None
    light_color: torch.Tensor | None = None
    point_light_weight: float | None = None
    light_intensity: float | None = None
    env_sh: torch.Tensor | None = None


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _resolve_image_path(root: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".png")
    options = [root / candidate, root / "images" / candidate]
    for option in options:
        if option.exists():
            return option
    raise FileNotFoundError(f"cannot resolve image {name} under {root}")


def _read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        color = file.readline().decode("ascii").strip() == "PF"
        width, height = map(int, file.readline().decode("ascii").split())
        scale = float(file.readline().decode("ascii"))
        dtype = "<f4" if scale < 0 else ">f4"
        channels = 3 if color else 1
        data = np.fromfile(file, dtype=dtype, count=width * height * channels)
    shape = (height, width, channels) if color else (height, width)
    return np.flipud(data.reshape(shape)).copy()


def _read_mvsnet_camera(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    extrinsic_index = lines.index("extrinsic")
    intrinsic_index = lines.index("intrinsic")
    extrinsic = torch.tensor(
        [[float(value) for value in lines[extrinsic_index + row + 1].split()] for row in range(4)]
    )
    intrinsic = torch.tensor(
        [[float(value) for value in lines[intrinsic_index + row + 1].split()] for row in range(3)]
    )
    return intrinsic.float(), torch.linalg.inv(extrinsic.float())


def _decompose_projection(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    projection = np.loadtxt(path).reshape(3, 4)
    # RQ via NumPy QR avoids adding a second OpenMP runtime on Windows.
    q, r = np.linalg.qr(np.flipud(projection[:, :3]).T)
    intrinsic = np.fliplr(np.flipud(r.T))
    rotation = np.flipud(q.T)
    signs = np.sign(np.diag(intrinsic))
    signs[signs == 0] = 1
    correction = np.diag(signs)
    intrinsic, rotation = intrinsic @ correction, correction @ rotation
    if np.linalg.det(rotation) < 0:
        intrinsic[:, 0] *= -1
        rotation[0] *= -1
    intrinsic /= intrinsic[2, 2]
    translation = np.linalg.solve(intrinsic, projection[:, 3])
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return (
        torch.from_numpy(intrinsic).float(),
        torch.linalg.inv(torch.from_numpy(world_to_camera)),
    )


def _read_binary_ply(
    path: Path, max_points: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    with path.open("rb") as file:
        header_lines = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"invalid PLY header: {path}")
            header_lines.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise NotImplementedError("only binary little-endian DTU PLY is supported")
        vertex_line = next(line for line in header_lines if line.startswith("element vertex"))
        vertex_count = int(vertex_line.split()[-1])
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.fromfile(file, dtype=dtype, count=vertex_count)
    indices = np.linspace(
        0, vertex_count - 1, min(max_points, vertex_count), dtype=np.int64
    )
    selected = vertices[indices]
    points = np.stack([selected["x"], selected["y"], selected["z"]], axis=-1)
    colors = np.stack(
        [selected["red"], selected["green"], selected["blue"]], axis=-1
    ).astype(np.float32) / 255
    return torch.from_numpy(points).float(), torch.from_numpy(colors).float()


class MultiViewScene:
    def __init__(
        self,
        frames: list[MultiViewFrame],
        points: torch.Tensor | None = None,
        point_colors: torch.Tensor | None = None,
    ):
        if not frames:
            raise ValueError("multiview scene has no frames")
        self.frames = frames
        self.points = points
        self.point_colors = point_colors

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> MultiViewFrame:
        return self.frames[index]

    @classmethod
    def from_blender(
        cls,
        root: str | Path,
        transforms: str = "transforms.json",
        scale: float = 1.0,
    ) -> "MultiViewScene":
        root = Path(root)
        meta = json.loads((root / transforms).read_text(encoding="utf-8"))
        frames: list[MultiViewFrame] = []
        for index, item in enumerate(meta["frames"]):
            path = _resolve_image_path(root, item["file_path"])
            with Image.open(path) as image:
                width, height = image.size
            width_s, height_s = round(width * scale), round(height * scale)
            if "fl_x" in item or "fl_x" in meta:
                fx = float(item.get("fl_x", meta["fl_x"])) * scale
                fy = float(item.get("fl_y", meta.get("fl_y", fx / scale))) * scale
            else:
                angle_x = float(meta["camera_angle_x"])
                fx = 0.5 * width_s / np.tan(0.5 * angle_x)
                fy = fx
            cx = float(item.get("cx", meta.get("cx", width / 2))) * scale
            cy = float(item.get("cy", meta.get("cy", height / 2))) * scale
            K = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
            )
            c2w_gl = torch.tensor(item["transform_matrix"], dtype=torch.float32)
            # Blender/OpenGL camera (-z forward, +y up) -> OpenCV.
            convert = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
            c2w = c2w_gl @ convert
            camera = PerspectiveCamera(K, c2w, width_s, height_s, index)
            diffuse_path = item.get("diffuse_path")
            diffuse = (
                _load_rgb(_resolve_image_path(root, diffuse_path), (width_s, height_s))
                if diffuse_path
                else None
            )
            env_sh = item.get("env_sh")
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width_s, height_s)),
                    camera=camera,
                    image_path=path,
                    diffuse_gt=diffuse,
                    light_position=(
                        torch.tensor(item["light_position"], dtype=torch.float32)
                        if "light_position" in item
                        else None
                    ),
                    light_direction=(
                        torch.tensor(item["light_direction"], dtype=torch.float32)
                        if "light_direction" in item
                        else None
                    ),
                    light_color=(
                        torch.tensor(item["light_color"], dtype=torch.float32)
                        if "light_color" in item
                        else None
                    ),
                    point_light_weight=(
                        float(item["point_light_weight"])
                        if "point_light_weight" in item
                        else None
                    ),
                    light_intensity=(
                        float(item["light_intensity"])
                        if "light_intensity" in item
                        else None
                    ),
                    env_sh=(
                        torch.tensor(env_sh, dtype=torch.float32).reshape(9, 3)
                        if env_sh is not None
                        else None
                    ),
                )
            )
        points_path = root / meta.get("points_path", "points.npy")
        colors_path = root / meta.get("point_colors_path", "point_colors.npy")
        points = (
            torch.from_numpy(np.load(points_path)).float()
            if points_path.exists()
            else None
        )
        colors = (
            torch.from_numpy(np.load(colors_path)).float()
            if colors_path.exists()
            else None
        )
        if colors is not None and colors.max() > 1:
            colors = colors / 255.0
        return cls(frames, points, colors)

    @classmethod
    def from_mvsnet(
        cls,
        root: str | Path,
        images_dir: str = "blended_images",
        cams_dir: str = "cams",
        depth_dir: str = "rendered_depth_maps",
        scale: float = 0.25,
        max_points: int = 50000,
        max_point_views: int = 16,
    ) -> "MultiViewScene":
        """Load an MVSNet/BlendedMVS scene and unproject its rendered depths."""
        root = Path(root)
        image_root, camera_root = root / images_dir, root / cams_dir
        image_paths = sorted(
            path
            for path in image_root.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and "_masked" not in path.stem
        )
        frames: list[MultiViewFrame] = []
        original_sizes: dict[int, tuple[int, int]] = {}
        for path in image_paths:
            camera_path = camera_root / f"{path.stem}_cam.txt"
            if not camera_path.exists():
                continue
            K, c2w = _read_mvsnet_camera(camera_path)
            with Image.open(path) as image:
                width, height = image.size
            width_s, height_s = round(width * scale), round(height * scale)
            K[0] *= width_s / width
            K[1] *= height_s / height
            frame_id = len(frames)
            original_sizes[frame_id] = (width, height)
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width_s, height_s)),
                    camera=PerspectiveCamera(K, c2w, width_s, height_s, frame_id),
                    image_path=path,
                )
            )
        if not frames:
            raise FileNotFoundError(f"no MVSNet image/camera pairs under {root}")

        points, colors = [], []
        depth_root = root / depth_dir
        view_ids = torch.linspace(
            0, len(frames) - 1, min(max_point_views, len(frames))
        ).long().unique().tolist()
        per_view = max(1, max_points // max(len(view_ids), 1))
        for frame_id in view_ids:
            frame = frames[frame_id]
            depth_path = depth_root / f"{frame.image_path.stem}.pfm"
            if not depth_path.exists():
                continue
            depth = torch.from_numpy(_read_pfm(depth_path)).float()
            if depth.ndim == 3:
                depth = depth[..., 0]
            valid = torch.isfinite(depth) & (depth > 1e-5)
            valid_ids = valid.flatten().nonzero().flatten()
            if not len(valid_ids):
                continue
            selected = valid_ids[
                torch.linspace(0, len(valid_ids) - 1, min(per_view, len(valid_ids))).long()
            ]
            height_d, width_d = depth.shape
            pixel_y = selected // width_d
            pixel_x = selected % width_d
            z = depth.flatten()[selected]
            K = frame.camera.K.clone()
            K[0] *= width_d / frame.camera.width
            K[1] *= height_d / frame.camera.height
            x = (pixel_x.float() - K[0, 2]) * z / K[0, 0]
            y = (pixel_y.float() - K[1, 2]) * z / K[1, 1]
            camera_points = torch.stack([x, y, z], dim=-1)
            rotation, translation = (
                frame.camera.c2w[:3, :3],
                frame.camera.c2w[:3, 3],
            )
            points.append(camera_points @ rotation.T + translation)
            grid = torch.stack(
                [
                    2 * pixel_x.float() / max(width_d - 1, 1) - 1,
                    2 * pixel_y.float() / max(height_d - 1, 1) - 1,
                ],
                dim=-1,
            ).view(1, -1, 1, 2)
            sampled = F.grid_sample(
                frame.image.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, :, :, 0].T
            colors.append(sampled)
        return cls(
            frames,
            torch.cat(points) if points else None,
            torch.cat(colors) if colors else None,
        )

    @classmethod
    def from_dtu(
        cls,
        root: str | Path,
        scan: int = 1,
        light: int = 3,
        scale: float = 0.125,
        max_points: int = 50000,
    ) -> "MultiViewScene":
        """Load a DTU SampleSet scan with official projection matrices."""
        root = Path(root)
        rectified = root / "Rectified" / f"scan{scan}"
        calibration = root / "Calibration" / "cal18"
        paths = sorted(rectified.glob(f"rect_*_{light}_r5000.png"))
        frames = []
        for path in paths:
            camera_number = int(path.stem.split("_")[1])
            projection_path = calibration / f"pos_{camera_number:03d}.txt"
            if not projection_path.exists():
                continue
            K, c2w = _decompose_projection(projection_path)
            with Image.open(path) as image:
                width, height = image.size
            width_s, height_s = round(width * scale), round(height * scale)
            K[0] *= width_s / width
            K[1] *= height_s / height
            frame_id = len(frames)
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width_s, height_s)),
                    camera=PerspectiveCamera(K, c2w, width_s, height_s, frame_id),
                    image_path=path,
                )
            )
        if not frames:
            raise FileNotFoundError(f"no DTU scan{scan} light{light} views under {root}")
        ply_path = root / "Points" / "stl" / f"stl{scan:03d}_total.ply"
        points, colors = (
            _read_binary_ply(ply_path, max_points)
            if ply_path.exists()
            else (None, None)
        )
        return cls(frames, points, colors)

    def fuse_projected_texture(
        self,
        points: torch.Tensor,
        frame_indices: list[int] | None = None,
        source_images: dict[int, torch.Tensor] | None = None,
        confidence_maps: dict[int, torch.Tensor] | None = None,
        occlusion_tolerance: float | None = None,
    ) -> torch.Tensor:
        """Project every 3D point into selected views and average RGB.

        The resulting persistent per-point texture is rendered into any target
        view and is the cross-view projected-texture condition for refinement.
        Restricting ``frame_indices`` prevents holdout inputs from leaking into
        train-time scene initialization.
        """
        points = points.cpu()
        color_sum = torch.zeros(len(points), 3)
        weight_sum = torch.zeros(len(points), 1)
        indices = (
            frame_indices if frame_indices is not None else list(range(len(self.frames)))
        )
        if not indices:
            raise ValueError("frame_indices must contain at least one view")
        for index in indices:
            frame = self.frames[index]
            uv, depth, valid = frame.camera.project(points)
            if occlusion_tolerance is not None:
                pixel_x = uv[:, 0].round().long().clamp(0, frame.camera.width - 1)
                pixel_y = uv[:, 1].round().long().clamp(0, frame.camera.height - 1)
                linear_index = pixel_y * frame.camera.width + pixel_x
                depth_buffer = torch.full(
                    (frame.camera.width * frame.camera.height,),
                    float("inf"),
                    dtype=depth.dtype,
                )
                depth_buffer.scatter_reduce_(
                    0,
                    linear_index[valid],
                    depth[valid],
                    reduce="amin",
                    include_self=True,
                )
                visible = depth <= (
                    depth_buffer[linear_index]
                    + occlusion_tolerance
                    * depth_buffer[linear_index].clamp(min=1)
                )
                valid = valid & visible
            gx = 2 * uv[:, 0] / max(frame.camera.width - 1, 1) - 1
            gy = 2 * uv[:, 1] / max(frame.camera.height - 1, 1) - 1
            grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
            source = (
                source_images[index]
                if source_images is not None and index in source_images
                else frame.image
            )
            sampled = F.grid_sample(
                source.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, :, :, 0].T
            weight = valid.float().unsqueeze(-1)
            if confidence_maps is not None and index in confidence_maps:
                sampled_confidence = F.grid_sample(
                    confidence_maps[index].unsqueeze(0),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0, :, :, 0].T
                weight = weight * sampled_confidence.clamp(0, 1)
            color_sum += sampled * weight
            weight_sum += weight
        return color_sum / weight_sum.clamp(min=1)

    @classmethod
    def from_colmap(
        cls,
        root: str | Path,
        sparse_dir: str = "sparse/0",
        images_dir: str = "images",
        scale: float = 1.0,
    ) -> "MultiViewScene":
        root = Path(root)
        sparse = root / sparse_dir
        cameras = _read_colmap_cameras(sparse / "cameras.txt", scale)
        image_records = _read_colmap_images(sparse / "images.txt")
        frames: list[MultiViewFrame] = []
        for frame_id, record in enumerate(image_records):
            camera_id, qvec, tvec, name = record
            K, width, height = cameras[camera_id]
            rotation = _qvec_to_rotation(qvec)
            w2c = torch.eye(4)
            w2c[:3, :3] = rotation
            w2c[:3, 3] = tvec
            c2w = torch.linalg.inv(w2c)
            path = root / images_dir / name
            camera = PerspectiveCamera(K, c2w, width, height, frame_id)
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width, height)),
                    camera=camera,
                    image_path=path,
                )
            )
        points, colors = _read_colmap_points(sparse / "points3D.txt")
        return cls(frames, points, colors)


def _read_colmap_cameras(
    path: Path, scale: float
) -> dict[int, tuple[torch.Tensor, int, int]]:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        camera_id, model = int(values[0]), values[1]
        width, height = round(int(values[2]) * scale), round(int(values[3]) * scale)
        params = [float(v) for v in values[4:]]
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = params[0] * scale
            cx, cy = params[1] * scale, params[2] * scale
        elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
            fx, fy = params[0] * scale, params[1] * scale
            cx, cy = params[2] * scale, params[3] * scale
        else:
            raise NotImplementedError(f"COLMAP camera model {model}")
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        cameras[camera_id] = (K, width, height)
    return cameras


def _read_colmap_images(
    path: Path,
) -> list[tuple[int, torch.Tensor, torch.Tensor, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        values = line.split()
        if len(values) >= 10:
            qvec = torch.tensor([float(v) for v in values[1:5]])
            tvec = torch.tensor([float(v) for v in values[5:8]])
            records.append((int(values[8]), qvec, tvec, values[9]))
            index += 2  # skip POINTS2D line
        else:
            index += 1
    return records


def _qvec_to_rotation(qvec: torch.Tensor) -> torch.Tensor:
    qvec = qvec / qvec.norm()
    w, x, y, z = qvec
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y]
            ),
            torch.stack(
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x]
            ),
            torch.stack(
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
            ),
        ],
    )


def _read_colmap_points(path: Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not path.exists():
        return None, None
    points, colors = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        points.append([float(v) for v in values[1:4]])
        colors.append([int(v) / 255.0 for v in values[4:7]])
    if not points:
        return None, None
    return torch.tensor(points, dtype=torch.float32), torch.tensor(colors)

