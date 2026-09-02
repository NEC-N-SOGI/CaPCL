import os
from pathlib import Path


def set_env_if_none(env_name: str, value: str) -> None:
    if env_name not in os.environ:
        os.environ[env_name] = value


def copy_env(src: str, dst: str) -> None:
    src_val = os.getenv(src, None)
    if src_val is None:
        return
    os.environ[dst] = src_val


def set_environment_variables(result_path: Path) -> None:
    # map OpenMPI launcher variables to torch.distributed ones
    copy_env("OMPI_COMM_WORLD_RANK", "RANK")
    copy_env("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK")
    copy_env("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE")

    set_env_if_none("HF_HOME", str(result_path / "hf_home"))
    set_env_if_none("TORCH_HOME", str(result_path / "torch_home"))
    set_env_if_none("HF_DATASETS_CACHE", str(result_path / "datasets_cache"))

    (Path(os.environ["HF_HOME"]) / "logs").mkdir(parents=True, exist_ok=True)
    (Path(os.environ["TORCH_HOME"]) / "logs").mkdir(parents=True, exist_ok=True)
    (Path(os.environ["HF_DATASETS_CACHE"]) / "logs").mkdir(
        parents=True, exist_ok=True
    )
