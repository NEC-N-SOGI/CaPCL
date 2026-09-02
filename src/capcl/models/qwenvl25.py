from typing import Any

import torch
from torchvision import transforms
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from capcl import registry
from capcl.dataset.imgtxt_dataset import BatchImgTxtOutput
from capcl.models.model import (
    BaseMixin,
    BaseModel,
    ModelBuilder,
    ModelOutput,
)

MODEL_NAMES = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
]


class QWenVL25(BaseModel):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        _model_name = f"Qwen/{model_name}"

        if not hasattr(Qwen2_5_VLForConditionalGeneration, "from_pretrained"):
            msg = (
                "Qwen2_5_VLForConditionalGeneration is not available. "
                "Please upgrade transformers to >=4.35.0."
            )
            raise ImportError(msg)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained(_model_name)

    def forward(
        self,
        batch: BatchImgTxtOutput,
        extract_gencap_feat: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> ModelOutput:
        dataset_names = batch["dataset_names"]

        gen_ids, _ = self.image_captioning(
            img_tensor=batch["img_tensor"],
            n_captions=self.N_GEN_CAPS,
        )

        dataset_name_tokens = self.processor.tokenizer(
            dataset_names,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(batch["img_tensor"].device)

        n_bs = batch["img_tensor"].shape[0]
        return ModelOutput(
            img_feat=torch.zeros(
                (n_bs, 100), device=batch["img_tensor"].device
            ),  # dummy tensor
            txt_feat=torch.zeros(
                (n_bs, 100), device=batch["img_tensor"].device
            ),  # dummy tensor
            gen_txt_feat=torch.zeros(
                (n_bs, 100), device=batch["img_tensor"].device
            ),  # dummy tensor
            dataset_names=dataset_name_tokens.input_ids.to(torch.int64),
            caption_tokens=gen_ids,
            new_text_label=torch.zeros(
                (n_bs, 5, 100), device=batch["img_tensor"].device, dtype=torch.int64
            ),  # dummy tensor
            new_text_logits=torch.zeros(
                (n_bs, 5, 100, 100), device=batch["img_tensor"].device
            ),  # dummy tensor
            gen_text_label=torch.zeros(
                (n_bs, 5, 100), device=batch["img_tensor"].device, dtype=torch.int64
            ),  # dummy tensor
        )

    def image_captioning(
        self,
        img_tensor: torch.Tensor,
        n_captions: int = 1,
        top_p: float = 0.95,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # for batched image captioning
        # generate text for each image in the batch
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": _img,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
            for _img in img_tensor
        ]
        texts = [
            self.processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        # Preparation for inference
        inputs = self.processor(
            text=texts,
            images=img_tensor.float(),
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        )
        inputs = inputs.to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(
            "cuda", dtype=torch.bfloat16
        )

        # Inference: Generation of the output with timeout protection
        # Force model to eval mode and disable compilation
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            top_p=top_p,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_captions,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_time=300.0,  # 5 minute timeout
            use_cache=True,
        )

        # Trim ids of the prompt text
        input_ids = inputs.input_ids.repeat(n_captions, 1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(input_ids, generated_ids, strict=True)
        ]

        n_bs = img_tensor.shape[0]
        tokens = torch.stack(generated_ids_trimmed, dim=0).view(n_bs, n_captions, -1)

        return tokens.contiguous(), torch.tensor([], device=img_tensor.device)


class QWenVL25Mixin(BaseMixin):
    img_size: tuple[int, int]

    def check_input_img_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if max(img_tensor.shape[-2:]) > max(self.img_size):
            msg = f"Input image size must be less than or equal to {self.img_size}. Got {img_tensor.shape[-2:]}"
            raise ValueError(msg)

        if img_tensor.dim() == 3:  # noqa: PLR2004 (3D tensor)
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor


@registry.register_model_builder("qwen_vl25")
class QWenVL25Builder(ModelBuilder):
    def __init__(
        self,
        model_class: str,  # noqa: ARG002
        model_type: str,
        target_module_name: str = "all",
    ) -> None:
        if model_type not in MODEL_NAMES:
            msg = f"Model type {model_type} is not supported. Choose from {MODEL_NAMES}."
            raise ValueError(msg)

        self._model = QWenVL25(model_type)

        to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(500, max_size=512)]
        )

        self._train_img_processor = to_tensor
        self._eval_img_processor = to_tensor

        self._txt_processor = self._model.processor.tokenizer
        self._model.tokenizer = self._model.processor.tokenizer
        self._model.temp = 1.0

        self._img_size = (512, 512)

        self._target_modules = self.get_all_modules()

        self._mixin = QWenVL25Mixin()
        self._mixin.img_size = self._img_size
        self._target_module_name = target_module_name

    def get_all_modules(self) -> list[str]:
        return [k for k, v in self._model.named_modules()]
