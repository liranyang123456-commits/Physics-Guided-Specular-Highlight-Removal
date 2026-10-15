"""Export the recovered 3D Gaussian geometry and BRDF parameters."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .scene import GaussianBRDFField


def export_scene(checkpoint: str, output_dir: str):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    colors = state.get("colors")
    field = GaussianBRDFField.from_point_cloud(state["points"], colors)
    # Extract only the nested scene state from the full pipeline.
    scene_state = {
        key.removeprefix("scene."): value
        for key, value in state["model"].items()
        if key.startswith("scene.")
    }
    field.load_state_dict(scene_state)
    field.eval()
    with torch.no_grad():
        material = field.materials()
        as_numpy = lambda tensor: tensor.detach().cpu().numpy()
        values = {
            "position": as_numpy(field.means),
            "scale": as_numpy(field.scales),
            "rotation_quaternion": as_numpy(
                torch.nn.functional.normalize(field.quaternions, dim=-1)
            ),
            "opacity": as_numpy(field.opacity),
            "normal": as_numpy(field.normals),
            "albedo": as_numpy(material.albedo),
            "specular": as_numpy(material.specular),
            "roughness": as_numpy(material.roughness),
            "metalness": as_numpy(material.metalness),
            "absorption": as_numpy(material.absorption),
            "projected_texture": as_numpy(field.texture),
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "gaussian_brdf.npz", **values)
    _write_ply(output / "gaussian_brdf.ply", values)
    print(f"exported 3D Gaussian BRDF model to {output}")


def _write_ply(path: Path, values: dict[str, np.ndarray]):
    count = len(values["position"])
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float opacity",
        "property float scale_x",
        "property float scale_y",
        "property float scale_z",
        "property float specular",
        "property float roughness",
        "property float metalness",
        "property float absorption",
        "end_header",
    ]
    lines = header
    rgb = (values["albedo"].clip(0, 1) * 255).astype(np.uint8)
    for index in range(count):
        p, n, c, s = (
            values["position"][index],
            values["normal"][index],
            rgb[index],
            values["scale"][index],
        )
        lines.append(
            f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]} "
            f"{c[0]} {c[1]} {c[2]} {values['opacity'][index, 0]} "
            f"{s[0]} {s[1]} {s[2]} {values['specular'][index, 0]} "
            f"{values['roughness'][index, 0]} {values['metalness'][index, 0]} "
            f"{values['absorption'][index, 0]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/world_export")
    args = parser.parse_args()
    export_scene(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

