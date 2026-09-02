import argparse
import json
from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw

from capcl import registry

SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (220, 50, 47),
    "green": (60, 160, 60),
    "blue": (50, 90, 200),
    "yellow": (220, 190, 40),
    "purple": (140, 80, 180),
    "orange": (230, 130, 40),
}
BACKGROUNDS = {
    "white": (245, 245, 245),
    "black": (25, 25, 25),
    "gray": (128, 128, 128),
    "beige": (222, 205, 175),
}
IMG_SIZE = 224
N_TRAIN = 16  # remaining combinations go to the test split


def draw_shape(shape: str, color_name: str, bg_name: str, idx: int) -> Image.Image:
    """Render one shape; position/size vary deterministically with idx."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BACKGROUNDS[bg_name])
    draw = ImageDraw.Draw(img)

    half = IMG_SIZE // 2
    radius = 40 + 8 * (idx % 5)
    cx = half + 20 * (idx % 3 - 1)
    cy = half + 20 * (idx // 3 % 3 - 1)
    color = COLORS[color_name]

    if shape == "circle":
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius), fill=color
        )
    elif shape == "square":
        draw.rectangle(
            (cx - radius, cy - radius, cx + radius, cy + radius), fill=color
        )
    else:  # triangle
        draw.polygon(
            [
                (cx, cy - radius),
                (cx - radius, cy + radius),
                (cx + radius, cy + radius),
            ],
            fill=color,
        )
    return img


def build_task(shape: str, data_root: Path, data_prefix: str) -> dict[str, dict]:
    """Create images and per-split dataset entries for one shape task."""
    img_dir = data_root / "synthetic_shapes" / shape
    img_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict] = {
        "train": {"images": [], "texts": []},
        "test": {"images": [], "texts": []},
    }
    combos = list(product(COLORS, BACKGROUNDS))
    for idx, (color_name, bg_name) in enumerate(combos):
        split = "train" if idx < N_TRAIN else "test"
        file_name = f"{split}_{idx:03d}.png"
        draw_shape(shape, color_name, bg_name, idx).save(img_dir / file_name)

        rel_path = f"{data_prefix}synthetic_shapes/{shape}/{file_name}"
        splits[split]["images"].append(rel_path)
        splits[split]["texts"].append(
            f"a {color_name} {shape} on a {bg_name} background"
        )

    return {
        split: {
            "name": f"synthetic_{shape}_{split}",
            "images": data["images"],
            "texts": data["texts"],
            # every caption matches exactly its own image
            "txt2img": [[i] for i in range(len(data["images"]))],
        }
        for split, data in splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=str, default="local")
    args = parser.parse_args()

    data_prefix, data_root = registry.get_data_path(args.machine)
    cfg_root = Path(registry.get_datacfg_root(args.machine))
    json_dir = cfg_root / "jsons"
    json_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg: dict[str, dict[str, str]] = {}
    for task_id, shape in enumerate(SHAPES, start=1):
        task = build_task(shape, Path(data_root), data_prefix)
        dataset_cfg[str(task_id)] = {"target_metric": "mrr"}
        for split, dataset in task.items():
            json_name = f"synthetic_{shape}_{split}.json"
            with (json_dir / json_name).open("w") as f:
                json.dump(dataset, f, indent=2)
            dataset_cfg[str(task_id)][split] = f"jsons/{json_name}"

    cfg_path = cfg_root / "synthetic_shapes.json"
    with cfg_path.open("w") as f:
        json.dump(dataset_cfg, f, indent=2)

    n_imgs = len(SHAPES) * len(COLORS) * len(BACKGROUNDS)
    print(f"Wrote {n_imgs} images to {data_root}synthetic_shapes/")
    print(f"Wrote dataset config to {cfg_path}")


if __name__ == "__main__":
    main()
