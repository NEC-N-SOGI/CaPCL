import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from peft import LoraModel, PeftModel
from transformers import PreTrainedTokenizerBase

from capcl.dataset import BatchImgTxtOutput

LORA_SUPPORTED_MODULES = [
    torch.nn.Linear,
    torch.nn.Embedding,
    torch.nn.modules.sparse.Embedding,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    transformers.pytorch_utils.Conv1D,
]  # peft.tuners.lora.model._create_new_module


@dataclass
class ModelOutput:
    """ModelOutput class for consistent output format.

    The model output is converted to a list of values in transformers.trainer.Trainer.prediction_step.
    (see transformers.trainer.py line 4473)

    This class (or SortDict) keeps the order of values in the converted list regardless of the initialization order.
    Metrics calculation (capcl.metrics) requires that the order of the output values must start with "loss", "txt_feat", and "img_feat".

    """

    img_feat: torch.Tensor
    txt_feat: torch.Tensor
    caption_tokens: torch.Tensor  # only for evaluation
    dataset_names: torch.Tensor
    new_text_label: torch.Tensor
    new_text_logits: torch.Tensor
    gen_txt_feat: torch.Tensor
    gen_text_label: torch.Tensor | None = None
    gen_text_logits: torch.Tensor | None = None
    densecap_text_label: torch.Tensor | None = None
    densecap_text_logits: torch.Tensor | None = None
    dense_txt_feats: torch.Tensor | None = None


class BaseMixin:
    img_size: tuple[int, int]

    def check_input_img_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if img_tensor.shape[-2:] != self.img_size:
            msg = f"Input image size must be {self.img_size}. Got {img_tensor.shape[-2:]}"
            raise ValueError(msg)

        if img_tensor.abs().max() <= 1.0:
            warnings.warn(
                "Maximum absolute value of input image tensor is less than 1.0. Check whether the input image tensor is normalized.",
                stacklevel=2,
            )

        if img_tensor.dim() == 3:  # noqa: PLR2004 (3D tensor)
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def prepare_input(
        self,
        batch: BatchImgTxtOutput,
    ) -> BatchImgTxtOutput:
        img_tensor: torch.Tensor = batch["img_tensor"]
        new_texts = batch["new_texts"]
        gen_caption = batch["gen_caption"]

        # check input image tensor: 1. shape, 2. normalization
        img_tensor = self.check_input_img_tensor(img_tensor)

        # check input texts: 1. number of images == number of texts?, 2. number of images == number of captions?
        if img_tensor.shape[0] != len(new_texts):
            msg = f"Length of texts must be {img_tensor.shape[0]}. Got {len(new_texts)}"
            raise ValueError(msg)

        if len(gen_caption) != len(new_texts):
            msg = f"Length of captions must be {len(new_texts)}. Got {len(gen_caption)}"
            raise ValueError(msg)

        return BatchImgTxtOutput(
            dataset_names=batch["dataset_names"],
            img_tensor=img_tensor,
            new_texts=new_texts,
            gen_caption=gen_caption,
            gen_txt_feat=batch["gen_txt_feat"],
            label_t2i=batch["label_t2i"],
            text_ids=batch["text_ids"],
            image_ids=batch["image_ids"],
            prev_img_tensor=batch["prev_img_tensor"],
            prev_text_tensor=batch["prev_text_tensor"],
            dense_cap=batch["dense_cap"],
        )


class BaseModel(torch.nn.Module):
    N_GEN_CAPS: int = 0
    return_all_word_tokens: bool = False
    temp: float | tuple[float, float]
    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    def forward(
        self,
        batch: BatchImgTxtOutput,
        **kwargs: Any,  # noqa: ANN401
    ) -> ModelOutput: ...

    # label variables are passed as kwargs. see transformers.trainer.py line 4432. note that label variables are in the batch.

    @abstractmethod
    def image_captioning(
        self, img_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def set_n_gen_caps(self, n_gen_caps: int) -> None:
        self.N_GEN_CAPS = n_gen_caps

    def set_return_all_word_tokens(self, return_all_word_tokens: bool) -> None:
        self.return_all_word_tokens = return_all_word_tokens

    def enable_return_all_word_tokens(self) -> None:
        self.set_return_all_word_tokens(True)

    def disable_return_all_word_tokens(self) -> None:
        self.set_return_all_word_tokens(False)


class ModelBuilder:
    @abstractmethod
    def __init__(
        self,
        *kargs: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        self._train_img_processor = lambda x: x
        self._eval_img_processor = lambda x: x
        self._txt_processor = lambda x: x
        self._img_size = (224, 224)
        self._target_modules: list[str] = []
        self._model: BaseModel
        self._mixin: BaseMixin
        self._target_module_name: str
        self._has_lm_head: bool

    @abstractmethod
    def get_target_modules(self, target_module_name: str) -> list[str]: ...

    @property
    def img_size(self) -> tuple[int, int]:
        return self._img_size

    @property
    def train_img_processor(self) -> Callable:
        return self._train_img_processor

    @property
    def eval_img_processor(self) -> Callable:
        return self._eval_img_processor

    @property
    def txt_processor(self) -> Callable:
        return self._txt_processor

    @property
    def target_modules(self) -> list[str]:
        return self._target_modules

    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def mixin(self) -> BaseMixin:
        return self._mixin

    @property
    def target_module_name(self) -> str:
        return self._target_module_name

    @property
    def has_lm_head(self) -> bool:
        return self._has_lm_head

    def enable_require_grad(self, parameters: Iterator) -> None:
        """Enable require_grad for the parameters."""
        for p in parameters:
            p.requires_grad = True

    def enable_all_modules(self, with_lora: bool = False) -> list[str]:
        target_modules = []
        for k, v in self.model.named_modules():
            req_grad = False
            if (type(v) in LORA_SUPPORTED_MODULES) or (not with_lora):
                target_modules.append(k)
                req_grad = True

            for p in v.parameters():
                p.requires_grad = req_grad

        return target_modules

    def disable_all_require_grad(self) -> None:
        # Freeze all modules by default
        for v in self.model.modules():
            for p in v.parameters():
                p.requires_grad = False


# This is a workaround to make PeftModel and LoraModel compatible with BaseModel.
class BasePeftLoraModel(BaseModel, PeftModel, LoraModel): ...  # type: ignore[misc,override]
