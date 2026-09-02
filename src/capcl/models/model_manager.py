import contextlib
import inspect
from pathlib import Path
from typing import Any, cast

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.pytorch_utils import Conv1D
from transformers.utils import logging

from capcl import registry
from capcl.models.model import BaseModel, BasePeftLoraModel
from capcl.tasks.task_cfg import LoRAConfig, ModelConfig

logger = logging.get_logger("transformers")


@registry.register_model_manager("ft")
class ModelManager:
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig | None = None,  # noqa: ARG002
    ) -> None:
        # model
        builder_class = registry.get_model_builder(model_config.model_class)
        cfg = model_config.model_dump()
        sig = inspect.signature(builder_class.__init__)
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            builder_kwargs = cfg
        else:
            allowed = set(sig.parameters.keys())
            allowed.discard("self")
            builder_kwargs = {k: v for k, v in cfg.items() if k in allowed}

        self.model_builder = builder_class(**builder_kwargs)

    def update_base_model(self) -> None:
        pass

    def from_pretrained(self, path: str) -> None:
        # Load the model state dict from a file
        if Path(path).is_file():
            file_path = Path(path)
        else:
            file_path = Path(path) / "pytorch_model.bin"

        state_dict = torch.load(str(file_path), map_location="cpu")
        self.model_builder.model.load_state_dict(state_dict)

    @property
    def base_model(self) -> BaseModel:
        return self.model_builder.model

    @property
    def model(self) -> BaseModel | BasePeftLoraModel:
        return self.model_builder.model

    def print_model_info(self) -> None:
        logger.info("[FT] Model Info: Fintuning the following modules")
        logger.info(self.model_builder.target_module_name)

    def disable_adapter(self) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def base_state_dict(self) -> dict[str, Any]:
        state_dict = self.model.state_dict()

        if not isinstance(state_dict, dict):
            msg = "state_dict is not dict"
            raise TypeError(msg)

        return state_dict

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer of the model."""
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer  # pyright: ignore[reportReturnType]
        msg = "Model does not have a tokenizer."
        raise AttributeError(msg)

    @property
    def pad_token_id(self) -> int:
        """Return the pad token id of the model."""
        _id = self.tokenizer.pad_token_id

        if isinstance(_id, int):
            return _id

        if isinstance(_id, list) and len(_id) == 1:
            return int(_id[0])

        if isinstance(_id, str):
            return int(_id)

        return -100

    @property
    def has_lm_head(self) -> bool:
        return bool(self.model_builder.has_lm_head)


@registry.register_model_manager("lora")
class LoraModelManager(ModelManager):
    # https://github.com/akashghimireOfficial/HuggingFace_101/blob/master/Turtorial/09_PEFT.ipynb
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig) -> None:
        super().__init__(model_config)
        self.merge_scale = lora_config.merge_scale

        target_modules = self._resolve_target_modules_for_lora(
            list(self.model_builder.target_modules)
        )
        if len(target_modules) == 0:
            msg = (
                "No LoRA-compatible target modules were resolved from model_builder.target_modules. "
                "Please set model_config.target_module_name to a value containing Linear/Embedding/Conv modules."
            )
            raise ValueError(msg)

        named_module_keys = {
            name for name, _ in self.model_builder.model.named_modules()
        }
        missing_targets = [
            module_name
            for module_name in target_modules
            if module_name not in named_module_keys
        ]
        if len(missing_targets) > 0:
            msg = (
                "Some configured target_modules were not found in model.named_modules(): "
                f"{missing_targets}"
            )
            raise ValueError(msg)

        # model: set LoraConfig
        config = LoraConfig(
            **lora_config.model_dump(exclude={"merge_scale"}),
            target_modules=target_modules,
            task_type=None,
        )
        # overwrap model with PEFT
        _model = cast("PreTrainedModel", self.model_builder.model)
        peft_model = get_peft_model(_model, config)

        if not isinstance(peft_model, PeftModel):
            msg = "Model is not PeftModel"
            raise TypeError(msg)

        self.peft_model = cast("BasePeftLoraModel", peft_model)  # pyright: ignore[reportAttributeAccessIssue]
        self.config = config

        target_module = self.model_builder.target_module_name
        if ("blip2" in model_config.model_class) and (
            "word" in target_module or target_module in ["qformer", "all"]
        ):
            # The following two parameters are same to each other.
            # After LoRA, these parameters become different.
            # To avoid this, we need to set the same value to them.
            # Thus, we make their lora params the same.

            # For BLIP2, we need to share the lora params
            # between query and text encoder.

            emb_a = "base_model.model.Qformer.bert.embeddings.word_embeddings.lora_embedding_A.default"
            emb_b = "base_model.model.Qformer.bert.embeddings.word_embeddings.lora_embedding_B.default"

            cls_a = "base_model.model.Qformer.cls.predictions.decoder.lora_A.default.weight"
            cls_b = "base_model.model.Qformer.cls.predictions.decoder.lora_B.default.weight"

            # share lora params between word embeddings and cls predictions
            self.share_lora_params(emb_b, cls_a)
            self.share_lora_params(emb_a, cls_b)

    def _resolve_target_modules_for_lora(
        self, configured_targets: list[str]
    ) -> list[str]:
        model = self.model_builder.model
        named_modules = dict(model.named_modules())
        supported_types = (
            torch.nn.Linear,
            torch.nn.Embedding,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.MultiheadAttention,
            Conv1D,
        )

        resolved: list[str] = []
        for target_name in configured_targets:
            prefix = f"{target_name}."
            for module_name, module in named_modules.items():
                if module_name != target_name and not module_name.startswith(prefix):
                    continue

                if isinstance(module, supported_types):
                    resolved.append(module_name)

        # preserve order and remove duplicates
        return list(dict.fromkeys(resolved))

    def share_lora_params(self, src_param: str, dst_param: str) -> None:
        parameter_dict = dict(self.peft_model.named_parameters())

        src = parameter_dict.get(src_param)
        dst = parameter_dict.get(dst_param)

        if not isinstance(src, torch.nn.Parameter) or not isinstance(
            dst, torch.nn.Parameter
        ):
            msg = f"{src_param} or {dst_param} is not a torch.nn.Parameter"
            raise TypeError(msg)

        dst.data = src.data.T.contiguous()
        dst.requires_grad = src.requires_grad

    def update_base_model(self) -> None:
        merge_scale = self.merge_scale

        for module in self.peft_model.modules():
            scaling = getattr(module, "scaling", None)

            if not isinstance(scaling, dict):
                continue

            for adapter_name, scale in list(scaling.items()):
                scaling[adapter_name] = scale * merge_scale  # ty: ignore[unsupported-operator]

        model: torch.nn.Module = self.peft_model.merge_and_unload()

        self.model_builder.model.load_state_dict(model.state_dict())

        _model = cast("PreTrainedModel", self.model_builder.model)

        peft_model = cast(
            "BasePeftLoraModel",
            get_peft_model(_model, self.config),
        )

        if not isinstance(peft_model, PeftModel):
            msg = "Model is not PeftModel"
            raise TypeError(msg)

        self.peft_model = peft_model  # pyright: ignore[reportAttributeAccessIssue]

    def from_pretrained(self, path: str) -> None:
        model = PeftModel.from_pretrained(self.base_model, path)
        self.peft_model = cast("BasePeftLoraModel", model)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def base_model(self) -> BaseModel:
        return self.model_builder.model

    @property
    def model(self) -> BaseModel | BasePeftLoraModel:
        return self.peft_model

    def disable_adapter(self) -> contextlib.AbstractContextManager:
        # a context manager that deactivates the adapter
        peft_model = self.peft_model
        return peft_model.disable_adapter()

    def print_model_info(self) -> None:
        logger.info("[LoRA] Model Info: Peft the following modules")
        logger.info(self.model_builder.target_module_name)
        logger.info("[Lora] Model Info: Active Adapters")
        logger.info(self.model.active_adapters)

        if isinstance(self.model, PeftModel):
            self.model.print_trainable_parameters()  # pyright: ignore[reportCallIssue]

    def base_state_dict(self) -> dict[str, Any]:
        state_dict = self.base_model.state_dict()

        if not isinstance(state_dict, dict):
            msg = "state_dict is not dict"
            raise TypeError(msg)

        # base_model.model.Qformer.bert.encoder.layer.0.attention.self.query.base_layer.weight

        for k, v in dict(state_dict).items():
            if "lora_" in k:
                del state_dict[k]

            if (not k.startswith("base_model.model.")) and ("base_layer" not in k):
                continue

            new_k = k.replace("base_model.model.", "")
            new_k = new_k.replace("base_layer.", "")

            state_dict[new_k] = v
            del state_dict[k]

        return state_dict
