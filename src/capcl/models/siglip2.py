from typing import TYPE_CHECKING, Any, cast

import torch
from PIL.Image import Image as PILImage
from transformers import AutoProcessor
from transformers.models.siglip.modeling_siglip import SiglipModel

from capcl.dataset import BatchImgTxtOutput
from capcl.models.model import (
    BaseMixin,
    BaseModel,
    ModelBuilder,
    ModelOutput,
)
from capcl.models.modules.normalize import (
    l2_normalization,
    wrap_model_layernorm_dtype_consistent,
)
from capcl.registry_class import registry

if TYPE_CHECKING:
    from transformers.models.siglip.image_processing_siglip import (
        SiglipImageProcessor,
    )
    from transformers.models.siglip.processing_siglip import SiglipProcessor

SIGLIP2_MODELS = {"giant": "google/siglip2-giant-opt-patch16-384"}

SIGLIP2_MODULES = [
    "all",
]


class SigLIP2(SiglipModel, BaseModel):
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        batch: BatchImgTxtOutput,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> ModelOutput:
        # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this

        image = cast("torch.FloatTensor", batch["img_tensor"])
        dtype = image.dtype

        # if tokens.shape[1] < 64, pad to 64,
        # since performance drops significantly without padding.
        tokens = torch.zeros(
            (image.size(0), 64),
            dtype=torch.int64,
            device=image.device,
        )
        # bs, n_texts, n_tokens --> bs, n_tokens
        tokens[:, : batch["new_texts"].size(-1)] = batch["new_texts"][:, 0]

        image_embeds_org = self.get_image_features(pixel_values=image)
        image_embeds_norm = l2_normalization(image_embeds_org).to(dtype)

        text_embeds_org = self.get_text_features(input_ids=tokens)
        text_embeds_norm = l2_normalization(text_embeds_org).to(dtype)

        dense_txt_feats = None
        if batch["dense_cap"].dim() > 1:
            features = []
            dense_caps = batch["dense_cap"]

            n_caps = dense_caps.shape[1]

            for i in range(n_caps):
                dense_tokens = torch.zeros(
                    (image.size(0), 64),
                    dtype=torch.int64,
                    device=image.device,
                )
                dense_tokens[:, : dense_caps.shape[-1]] = dense_caps[:, i]

                text_embeds = self.get_text_features(input_ids=dense_tokens)

                text_embeds_normalized = text_embeds / text_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )
                # bs, feat_dim
                features.append(text_embeds_normalized.unsqueeze(1))
            # bs, n_caps, feat_dim
            dense_txt_feats = torch.stack(features, dim=1)

        dataset_name_tokens = self.tokenizer(
            batch["dataset_names"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(image.device)

        return ModelOutput(
            img_feat=image_embeds_norm,
            txt_feat=text_embeds_norm,
            gen_txt_feat=torch.tensor([], device=image.device),
            dataset_names=dataset_name_tokens.input_ids.to(torch.int64),
            caption_tokens=torch.tensor([], device=image.device),
            new_text_label=torch.tensor([], device=image.device),
            new_text_logits=torch.tensor([], device=image.device),
            gen_text_label=None,
            gen_text_logits=None,
            densecap_text_label=None,
            densecap_text_logits=None,
            dense_txt_feats=dense_txt_feats,
        )

    # label variables are passed as kwargs. see transformers.trainer.py line 4432. note that label variables are in the batch.

    def image_captioning(
        self, img_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        msg = "SigLIP2 does not support image captioning."
        raise NotImplementedError(msg)


@registry.register_model_builder("siglip2")
class SigLIP2Builder(ModelBuilder):
    def __init__(
        self,
        model_class: str,  # noqa: ARG002
        model_type: str,
        target_module_name: str,
        **kwargs: Any,  # noqa: ARG002,ANN401
    ) -> None:
        self._has_lm_head = False
        if model_type not in SIGLIP2_MODELS:
            msg = f"model_type must be one of {SIGLIP2_MODELS}. Got {model_type}"
            raise ValueError(msg)

        model_name = SIGLIP2_MODELS[model_type]

        _model = SigLIP2.from_pretrained(
            model_name,
            device_map="cpu",
            attn_implementation="sdpa",
        )

        processor: SiglipProcessor = AutoProcessor.from_pretrained(model_name)

        img_processor: SiglipImageProcessor = processor.image_processor  # pyright: ignore[reportAttributeAccessIssue]
        txt_processor = processor.tokenizer  # pyright: ignore[reportAttributeAccessIssue]

        _model.to(torch.bfloat16)  # pyright: ignore[reportArgumentType]
        _model.tokenizer = txt_processor

        logit_scale, logit_bias = (
            _model.logit_scale.exp().item(),
            _model.logit_bias.item(),
        )
        _model.temp = (logit_scale, logit_bias)

        wrap_model_layernorm_dtype_consistent(_model)

        self._model = _model
        self.img_processor = img_processor
        self._img_size = (img_processor.size["height"], img_processor.size["width"])

        txt_processor.max_len = 64
        self._txt_processor = txt_processor

        self._mixin = BaseMixin()
        self._mixin.img_size = self._img_size

        self._target_modules = self.get_target_modules(target_module_name)
        self._target_module_name = target_module_name

    def _train_img_processor(self, img: PILImage) -> torch.Tensor:
        return torch.from_numpy(self.img_processor(img)["pixel_values"][0])

    def _eval_img_processor(self, img: PILImage) -> torch.Tensor:
        return torch.from_numpy(self.img_processor(img)["pixel_values"][0])

    def get_target_modules(self, target_module_name: str) -> list[str]:
        if target_module_name not in SIGLIP2_MODULES:
            msg = f"target_module_name must be one of {SIGLIP2_MODULES}. Got {target_module_name}"
            raise ValueError(msg)

        return self.enable_all_modules()
