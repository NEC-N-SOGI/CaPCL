from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from capcl.loss.sim_distill import SimDistillLoss
    from capcl.models.model import ModelBuilder
    from capcl.models.model_manager import ModelManager

T = TypeVar("T")


class Mapper(NamedTuple):
    model_builders: dict[str, type[ModelBuilder]]
    model_managers: dict[str, type[ModelManager]]
    sim_distill_loss: dict[str, type[SimDistillLoss]]
    result_path: dict[str, tuple[str, str]]
    data_path: dict[str, tuple[str, str]]
    datacfg_root: dict[str, str]


class Registry:
    mapping = Mapper({}, {}, {}, {}, {}, {})

    @classmethod
    def _register(
        cls, mapping_dict: dict[str, type[T]], name: str
    ) -> Callable[[type[T]], type[T]]:
        def _register_decorator(target_class: type[T]) -> type[T]:
            if name in mapping_dict:
                msg = f"{name} is already registered in {mapping_dict=}"
                raise KeyError(msg)

            mapping_dict[name] = target_class
            return target_class

        return _register_decorator

    @classmethod
    def register_model_builder(
        cls, name: str
    ) -> Callable[[type[ModelBuilder]], type[ModelBuilder]]:
        return cls._register(cls.mapping.model_builders, name)

    @classmethod
    def register_model_manager(
        cls, name: str
    ) -> Callable[[type[ModelManager]], type[ModelManager]]:
        return cls._register(cls.mapping.model_managers, name)

    @classmethod
    def register_sim_distill_loss(
        cls, name: str
    ) -> Callable[[type[SimDistillLoss]], type[SimDistillLoss]]:
        return cls._register(cls.mapping.sim_distill_loss, name)

    @classmethod
    def register_result_path(cls, machine: str, path: str) -> None:
        if machine in cls.mapping.result_path:
            msg = f"{machine} is already registered in result_path"
            raise KeyError(msg)
        cls.mapping.result_path[machine] = ("/path-to-results/", path)

    @classmethod
    def register_data_path(cls, machine: str, path: str) -> None:
        if machine in cls.mapping.data_path:
            msg = f"{machine} is already registered in data_path"
            raise KeyError(msg)
        cls.mapping.data_path[machine] = ("/path-to-data-dir/", path)

    @classmethod
    def register_datacfg_root(cls, machine: str, path: str) -> None:
        if machine in cls.mapping.datacfg_root:
            msg = f"{machine} is already registered in datacfg_root"
            raise KeyError(msg)
        cls.mapping.datacfg_root[machine] = path

    # Getters for model builders
    @classmethod
    def get_model_builder(cls, name: str) -> type[ModelBuilder]:
        if name not in cls.mapping.model_builders:
            msg = f"{name} is not registered in model_builders"
            raise KeyError(msg)
        return cls.mapping.model_builders[name]

    # Getters for model managers
    @classmethod
    def get_model_manager(cls, name: str) -> type[ModelManager]:
        if name not in cls.mapping.model_managers:
            msg = f"{name} is not registered in model_managers"
            raise KeyError(msg)
        return cls.mapping.model_managers[name]

    # Getters for sim distill loss
    @classmethod
    def get_sim_distill_loss(cls, name: str) -> type[SimDistillLoss]:
        if name not in cls.mapping.sim_distill_loss:
            msg = f"{name} is not registered in sim_distill_loss"
            raise KeyError(msg)
        return cls.mapping.sim_distill_loss[name]

    # Getters for paths
    @classmethod
    def get_data_path(cls, machine: str) -> tuple[str, str]:
        if machine not in cls.mapping.data_path:
            msg = f"{machine} is not registered in data_path"
            raise KeyError(msg)
        return cls.mapping.data_path[machine]

    @classmethod
    def get_result_path(cls, machine: str) -> tuple[str, str]:
        if machine not in cls.mapping.result_path:
            msg = f"{machine} is not registered in result_path"
            raise KeyError(msg)
        return cls.mapping.result_path[machine]

    @classmethod
    def get_datacfg_root(cls, machine: str) -> str:
        if machine not in cls.mapping.datacfg_root:
            msg = f"{machine} is not registered in datacfg_root"
            raise KeyError(msg)
        return cls.mapping.datacfg_root[machine]


registry = Registry()

file_directory = Path(__file__).parent

with (file_directory / "directory_settings.json").open("r") as f:
    directory_settings = json.load(f)

for k, v in directory_settings["result_path"].items():
    registry.register_result_path(k, v)

for k, v in directory_settings["data_path"].items():
    registry.register_data_path(k, v)

for k, v in directory_settings["config_path"].items():
    registry.register_datacfg_root(k, v)
