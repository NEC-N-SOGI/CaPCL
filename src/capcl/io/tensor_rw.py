import hashlib
from pathlib import Path

import torch
from filelock import FileLock
from safetensors.torch import load_file, save_file


def safe_save(tensor_dict: dict[str, torch.Tensor], path: Path) -> None:
    temp_filename = None
    temp_dir = None
    try:
        hash_str = hashlib.sha256(str(path).encode()).hexdigest()
        temp_dir = path.parent / f".temp_{hash_str}"
        temp_dir.mkdir(exist_ok=True)

        lock_file = temp_dir / "lockfile"
        temp_filename = temp_dir / f"{path.name}.temp"

        with FileLock(lock_file):
            save_file(tensor_dict, temp_filename)

            temp_filename.rename(path)

    except Exception:
        if temp_filename and temp_filename.exists():
            temp_filename.unlink(missing_ok=True)
        raise

    finally:
        if temp_dir and temp_dir.exists():
            for file in temp_dir.iterdir():
                file.unlink()
            temp_dir.rmdir()


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    _tensor = tensor.cpu()
    safe_save({"tensor": _tensor}, path)


def load_tensor(path: Path) -> torch.Tensor:
    return load_file(path)["tensor"]


def save_tensor_dict(tensor_dict: dict[str, torch.Tensor], path: Path) -> None:
    tensor_dict_cpu = {k: v.cpu() for k, v in tensor_dict.items()}
    safe_save(tensor_dict_cpu, path)


def load_tensor_dict(path: Path) -> dict[str, torch.Tensor]:
    return load_file(path)
