import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

from capcl import registry
from capcl.dataset import DatasetDictModel, DatasetModel
from capcl.io.tensor_rw import load_tensor
from capcl.metrics import CaptionCache


@dataclass
class ImgTxtOutput:
    img_tensor: torch.Tensor
    new_text: (
        torch.Tensor
    )  # new concept only. used to ITC or other conventional loss.
    gen_caption: torch.Tensor  # new concept with generated caption. used to lm_loss. if not specified, same as text.
    gen_txt_feat: torch.Tensor  # generated text feature for ITC loss.
    label_t2i: torch.Tensor
    text_id: int
    image_id: int
    dataset_name: str
    dense_cap: torch.Tensor  # dense caption tokens, if available
    prev_img_tensor: torch.Tensor | None = None
    prev_text_tensor: torch.Tensor | None = None


class BatchImgTxtOutput(TypedDict):
    img_tensor: torch.Tensor
    new_texts: torch.Tensor
    gen_caption: torch.Tensor
    gen_txt_feat: torch.Tensor
    label_t2i: torch.Tensor
    text_ids: torch.Tensor
    image_ids: torch.Tensor
    dataset_names: list[str]
    dense_cap: torch.Tensor
    prev_img_tensor: torch.Tensor | None
    prev_text_tensor: torch.Tensor | None


class ImgTxtCollator(DataCollatorMixin):
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def stack_text_tokens(self, token_list: list[torch.Tensor]) -> torch.Tensor:
        # token_list[i].shape = (n_tokens, dim).
        # n_tokens are different for each image.
        # stack them to (n_images, n_max_tokens, dim)

        if all(i.shape[0] == 0 for i in token_list):
            # all tokens are empty
            return torch.empty(len(token_list), dtype=token_list[0].dtype).to(
                token_list[0].device
            )

        max_len = max([i.shape[0] for i in token_list])
        max_dim = max([i.shape[1] for i in token_list])

        # pad to max_len
        padded_tokens = torch.zeros(
            (len(token_list), max_len, max_dim), dtype=token_list[0].dtype
        ).to(token_list[0].device)
        padded_tokens.fill_(self.pad_token_id)

        for i, tokens in enumerate(token_list):
            padded_tokens[i, : tokens.shape[0], : tokens.shape[1]] = tokens

        return padded_tokens

    def stack_images_with_padding(
        self, img_tensor: list[torch.Tensor]
    ) -> torch.Tensor:
        # stack them to (n_images, 3, H, W) with padding to max H and W

        max_h = max([i.shape[-2] for i in img_tensor])
        max_w = max([i.shape[-1] for i in img_tensor])

        padded_images = torch.zeros(
            (len(img_tensor), 3, max_h, max_w), dtype=img_tensor[0].dtype
        ).to(img_tensor[0].device)

        for i, img in enumerate(img_tensor):
            padded_images[i, :, : img.shape[-2], : img.shape[-1]] = img

        return padded_images

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        features: list[ImgTxtOutput],
        return_tenbsors: None | bool = None,  # noqa: ARG002
    ) -> BatchImgTxtOutput:
        img_tensor = self.stack_images_with_padding([f.img_tensor for f in features])

        label_t2i = torch.stack([f.label_t2i for f in features])
        text_ids = torch.tensor([f.text_id for f in features])
        image_ids = torch.tensor([f.image_id for f in features])

        texts = self.stack_text_tokens([f.new_text for f in features])
        caption = self.stack_text_tokens([f.gen_caption for f in features])
        gen_txt_feat = self.stack_text_tokens([f.gen_txt_feat for f in features])

        dense_cap = self.stack_text_tokens([f.dense_cap for f in features])

        dataset_names = [f.dataset_name for f in features]

        if any(f.prev_img_tensor is None for f in features) or any(
            f.prev_text_tensor is None for f in features
        ):
            prev_img_tensor = None
            prev_text_tensor = None
        else:
            prev_img_tensor = torch.stack(
                [
                    f.prev_img_tensor
                    for f in features
                    if f.prev_img_tensor is not None
                ]
            )
            prev_text_tensor = torch.stack(
                [
                    f.prev_text_tensor
                    for f in features
                    if f.prev_text_tensor is not None
                ]
            )

        return BatchImgTxtOutput(
            img_tensor=img_tensor,
            new_texts=texts,
            gen_caption=caption,
            gen_txt_feat=gen_txt_feat,
            label_t2i=label_t2i,
            text_ids=text_ids,
            image_ids=image_ids,
            dataset_names=dataset_names,
            prev_img_tensor=prev_img_tensor,
            prev_text_tensor=prev_text_tensor,
            dense_cap=dense_cap,
        )


def cvt_txt2img_to_img2txt(txt2img: list[list[int]], n_img: int) -> list[list[int]]:
    return [
        [j for j in range(len(txt2img)) if i in txt2img[j]] for i in range(n_img)
    ]


class ImgTxtDataset(Dataset):
    def __init__(
        self,
        dataset: DatasetModel,
        img_processor: Callable,
        machine: str,
        tokenizer: PreTrainedTokenizerBase,
        caption_dir: Path | None = None,
        remove_multi_texts: bool = False,
        has_non_targeted_images: bool = False,
        use_caption: bool = False,
        use_prev_feat: bool = False,
        max_length: int = 128,
        captioning_model: str = "",
    ) -> None:
        if use_caption and tokenizer is None:
            msg = "tokenizer must be specified when use_caption is True"
            raise ValueError(msg)
        self.dataset = dataset

        self.img_processor = img_processor
        self.txt2img = self.dataset.txt2img
        self.tokenizer = tokenizer
        self.use_caption = use_caption
        self.use_prev_feat = use_prev_feat
        self.max_length = max_length

        self.img_txt_pairs = self._gen_img_txt_pairs(has_non_targeted_images)
        img_ids = list({img_id for img_id, _ in self.img_txt_pairs})
        txt_ids = list({txt_id for _, txt_id in self.img_txt_pairs})
        self._check_img_ids(img_ids)

        self.caption_dir = caption_dir
        self.caption_paths = self._get_caption_token_paths(caption_dir, img_ids)
        self.text_feat_paths = self._get_caption_token_paths(caption_dir, txt_ids)

        self.data_prefix, self.data_root = registry.get_data_path(machine)

        self.img_ids = img_ids
        self.txt_ids = txt_ids

        self.captioning_model = captioning_model

        if not remove_multi_texts:
            return

        valid_pair_ids = []
        registered_img_ids = set()
        for i, (img_id, _) in enumerate(self.img_txt_pairs):
            if img_id in registered_img_ids:
                continue

            valid_pair_ids.append(i)
            registered_img_ids.add(img_id)

        self.img_txt_pairs = [self.img_txt_pairs[i] for i in valid_pair_ids]

    def __len__(self) -> int:
        return len(self.img_txt_pairs)

    def _check_img_ids(self, img_ids: list[int]) -> None:
        if any(img_id >= len(self.dataset.images) for img_id in img_ids):
            msg = (
                f"img_id must be less than the number of images. "
                f"img_id: {len(img_ids)}, num_images: {len(self.dataset.images)}"
            )
            raise ValueError(msg)

        if len(set(img_ids)) != len(self.dataset.images):
            diff = set(range(len(self.dataset.images))) - set(img_ids)
            diff_str = ", ".join(map(str, diff))
            warnings.warn(
                f"[{self.dataset.name}]Some images are not targeted by the text. Missing ids: {diff_str}",
                stacklevel=2,
            )

    def _gen_img_txt_pairs(
        self,
        has_non_targeted_images: bool,
    ) -> list[tuple[int, int]]:
        txt_img_pairs = [
            (img_id, text_id)
            for text_id, txt2img in enumerate(self.txt2img)
            for img_id in txt2img
        ]
        if not has_non_targeted_images:
            return txt_img_pairs

        targeted_images = {img_id for img_id, _ in txt_img_pairs}
        non_targeted_images = set(range(len(self.dataset.images))) - targeted_images

        non_targeted_pairs = [(img_id, -1) for img_id in non_targeted_images]

        return txt_img_pairs + non_targeted_pairs

    def _get_caption_token_paths(
        self, caption_dir: Path | None, img_ids: list[int]
    ) -> dict[int, str]:
        caption_paths: dict[int, str] = {}

        if caption_dir is None:
            return caption_paths

        caption_dir = Path(caption_dir)

        for img_id in img_ids:
            binpath = CaptionCache.imgid_to_binpath(img_id, caption_dir)

            caption_paths[img_id] = binpath.absolute().as_posix()

        return caption_paths

    def _prepare_img(self, idx: int) -> tuple[int, str, torch.Tensor, torch.Tensor]:
        img_id, _ = self.img_txt_pairs[idx]
        img_path = self.dataset.images[img_id].replace(
            self.data_prefix, self.data_root
        )
        img = Image.open(img_path)
        # to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width < 10 or img.height < 10:  # noqa: PLR2004
            img = img.resize((100, 100))

        if self.captioning_model == "":
            return img_id, img_path, self.img_processor(img), torch.empty(0)

        cap_path = (
            Path(img_path).parent.parent
            / self.captioning_model
            / f"{Path(img_path).parent.name}-{Path(img_path).name}"
        )
        with cap_path.with_suffix(".txt").open("r") as f:
            captions = f.readlines()

        all_tokens = []
        for target_cap in captions:
            tokenized = self.tokenizer(
                target_cap.strip(),
                return_tensors="pt",
                max_length=self.max_length,
                truncation=False,
                padding="max_length",
            )
            tokens = tokenized.input_ids.squeeze()
            all_tokens.append(tokens[: self.max_length])

        return img_id, img_path, self.img_processor(img), torch.stack(all_tokens)

    def _select_text(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        _, text_id = self.img_txt_pairs[idx]

        if text_id == -1:
            return text_id, torch.empty(0), torch.zeros(len(self.dataset.images))

        t2i_label_idxs: list[int] = self.txt2img[text_id]
        t2i_label_onehot = torch.zeros(len(self.dataset.images))
        for i in t2i_label_idxs:
            t2i_label_onehot[i] = 1

        text = self.dataset.texts[text_id]

        tokenized = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, truncation=True
        )

        return text_id, tokenized.input_ids, t2i_label_onehot

    def __getitem__(self, idx: int) -> ImgTxtOutput:  # ty: ignore[invalid-method-override]
        img_id, _, img_tensor, dense_cap = self._prepare_img(idx)

        text_id, text, t2i_label_onehot = self._select_text(idx)

        # select caption of the image
        # if there are multiple captions, randomly select one

        if self.use_caption:
            token_bin_path = self.caption_paths[img_id]

            # generated caption
            tokens = load_tensor(Path(token_bin_path))
            gen_captions = torch.tensor(tokens)

            # generated caption feat
            gen_txt_feat = load_tensor(
                Path(token_bin_path).with_suffix(".gen_feat_bin")
            ).to(torch.bfloat16)

            if gen_txt_feat.dim() > 2:  # noqa: PLR2004
                # [n_caps, n_tokens, dim] -> [n_caps, dim]
                gen_txt_feat = gen_txt_feat[:, 0, :]
        else:
            gen_captions = torch.empty(0)
            gen_txt_feat = torch.empty(0)

        if self.use_prev_feat:
            token_bin_path = self.caption_paths[img_id]
            # prev text and img feats
            bin_path = Path(token_bin_path)
            image_bin_path = (
                bin_path.parent / ("img_feat_" + bin_path.name)
            ).absolute()

            token_bin_path = self.text_feat_paths[text_id]
            bin_path = Path(token_bin_path)
            text_bin_path = (
                bin_path.parent / ("text_feat_" + bin_path.name)
            ).absolute()

            prev_text_feat = load_tensor(text_bin_path).to(torch.bfloat16)
            prev_img_feat = load_tensor(image_bin_path).to(torch.bfloat16)

            if prev_text_feat.dim() > 1:
                # [n_tokens, dim] -> [dim]
                prev_text_feat = prev_text_feat[0].squeeze()
        else:
            prev_text_feat = None
            prev_img_feat = None

        return ImgTxtOutput(
            dataset_name=self.dataset.name,
            img_tensor=img_tensor,
            new_text=text,
            gen_caption=gen_captions,
            gen_txt_feat=gen_txt_feat,
            label_t2i=t2i_label_onehot,
            text_id=text_id,
            image_id=img_id,
            prev_img_tensor=prev_img_feat,
            prev_text_tensor=prev_text_feat,
            dense_cap=dense_cap,  # dense caption tokens, if available
        )


class DatasetBuilder:
    @staticmethod
    def build(
        dataset_path: str | Path,
        caption_path_dict: dict[str, Path] | None,
        img_processor: Callable,
        machine: str,
        tokenizer: PreTrainedTokenizerBase,
        split_prefix: str = "",
        has_non_targeted_images: bool = False,
        use_caption: bool = False,
        use_prev_feat: bool = False,
        captioning_model: str = "",
    ) -> dict[str, ImgTxtDataset]:
        datasets = DatasetDictModel.from_json(dataset_path)

        dataset_dict = {}
        _caption_path_dict = caption_path_dict or {}
        for split in datasets.dataset:
            if (not split.startswith(split_prefix)) and (split_prefix != ""):
                continue
            dataset = datasets.dataset[split]
            dataset_dict[split] = ImgTxtDataset(
                dataset=dataset,
                caption_dir=_caption_path_dict.get(split, None),
                img_processor=img_processor,
                machine=machine,
                has_non_targeted_images=has_non_targeted_images,
                use_caption=use_caption,
                use_prev_feat=use_prev_feat,
                tokenizer=tokenizer,
                captioning_model=captioning_model,
                max_length=tokenizer.max_len,  # pyright: ignore[reportArgumentType]
            )

        return dataset_dict

    @staticmethod
    def get_training_dataset_keys(datasets_path: str | Path) -> list[str]:
        datasets = DatasetDictModel.from_json(datasets_path)
        return natsorted([k for k in datasets.dataset if k.startswith("train")])

    @staticmethod
    def build_from_split(
        dataset_path: str | Path,
        img_processor: Callable,
        split: str,
        machine: str,
        tokenizer: PreTrainedTokenizerBase,
        caption_path: Path | None = None,
        remove_multi_texts: bool = False,
        has_non_targeted_images: bool = False,
        use_caption: bool = False,
        use_prev_feat: bool = False,
        captioning_model: str = "",
    ) -> ImgTxtDataset:
        datasets = DatasetDictModel.from_json(dataset_path)
        dataset = datasets.dataset[split]
        return ImgTxtDataset(
            dataset=dataset,
            caption_dir=caption_path,
            img_processor=img_processor,
            remove_multi_texts=remove_multi_texts,
            tokenizer=tokenizer,
            machine=machine,
            has_non_targeted_images=has_non_targeted_images,
            use_caption=use_caption,
            use_prev_feat=use_prev_feat,
            captioning_model=captioning_model,
            max_length=tokenizer.max_len,  # pyright: ignore[reportArgumentType]
        )

    @staticmethod
    def train_build(
        datasets_path: str | Path,
        img_processor: Callable,
        machine: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, ImgTxtDataset]:
        return DatasetBuilder.build(
            dataset_path=datasets_path,
            caption_path_dict=None,
            img_processor=img_processor,
            machine=machine,
            tokenizer=tokenizer,
            split_prefix="train",
            has_non_targeted_images=True,
            use_caption=False,
            use_prev_feat=False,
        )

    @staticmethod
    def eval_build(
        datasets_path: str | Path,
        img_processor: Callable,
        machine: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, ImgTxtDataset]:
        dataset_dict_val = DatasetBuilder.build(
            dataset_path=datasets_path,
            img_processor=img_processor,
            caption_path_dict=None,
            split_prefix="val",
            machine=machine,
            has_non_targeted_images=True,
            use_caption=False,
            use_prev_feat=False,
            tokenizer=tokenizer,
        )

        dataset_dict_test = DatasetBuilder.build(
            dataset_path=datasets_path,
            img_processor=img_processor,
            caption_path_dict=None,
            split_prefix="test",
            machine=machine,
            has_non_targeted_images=True,
            use_caption=False,
            use_prev_feat=False,
            tokenizer=tokenizer,
        )

        return {**dataset_dict_val, **dataset_dict_test}
