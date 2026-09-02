from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from capcl.io.tensor_rw import save_tensor

if TYPE_CHECKING:
    import numpy as np
    from transformers import EvalPrediction, PreTrainedTokenizerBase

    class EvalPredictionFlexible(EvalPrediction):
        # In the original EvalPrediction, predictions and label_ids are annotated as tuple[np.ndarray]
        # But, in fact, they can be tuples of np.ndarray with flexible length.
        # Here we redefine them to avoid linting errors.
        predictions: tuple[np.ndarray, ...]
        label_ids: tuple[np.ndarray, ...]


class CaptionCache:
    def __init__(
        self,
        output_dir: Path,
        tokenizer: PreTrainedTokenizerBase | None = None,
        add_dataset_name: bool = False,
        save_features: bool = True,
        img_names: list[str] | None = None,
        use_parent_as_name: bool = False,
    ) -> None:
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.add_dataset_name = add_dataset_name
        self.save_features = save_features
        self.img_names = img_names
        self.use_parent_as_name = use_parent_as_name

    @staticmethod
    def imgid_to_binpath(img_id: int, output_dir: Path) -> Path:
        return output_dir / (f"{img_id:08d}" + ".bin")

    @staticmethod
    def expand_eval_prediction(
        res: EvalPredictionFlexible,
    ) -> tuple[
        torch.Tensor,
        list[int],
        list[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # res.label_ids = (label_t2i, text_ids, image_ids) defined by transformers.TrainingArguments.label_names.

        if len(res.label_ids) < 3:  # noqa: PLR2004
            msg = f"label_ids must have at least 3 elements (label_t2i, text_ids, image_ids), but got {len(res.label_ids)}"
            raise IndexError(msg)

        if len(res.predictions) < 3:  # noqa: PLR2004
            msg = f"predictions must have at least 3 elements (text_feat, img_feat, caption_tokens), but got {len(res.predictions)}"
            raise IndexError(msg)

        text_ids_list = res.label_ids[1].tolist()

        img_ids_list = [int(i) for i in res.label_ids[2]]

        text_feat = torch.tensor(res.predictions[0], dtype=torch.bfloat16)
        text_feat[text_feat <= -1e2] = 0.0  # noqa: PLR2004

        img_feat = torch.tensor(res.predictions[1], dtype=torch.bfloat16)

        caption_tokens = torch.tensor(res.predictions[2], dtype=torch.int64).to(
            torch.int64
        )

        dataset_name_tokens = torch.tensor(res.predictions[3], dtype=torch.int64)

        gen_txt_feat = torch.tensor(res.predictions[4], dtype=torch.bfloat16)

        return (
            caption_tokens,
            text_ids_list,
            img_ids_list,
            text_feat,
            img_feat,
            dataset_name_tokens,
            gen_txt_feat,
        )

    def imgid_to_imgpath(
        self,
        img_ids: list[int],
        dataset_name: torch.Tensor,
        names: list[str] | None = None,
    ) -> list[Path]:
        if self.add_dataset_name:
            if self.tokenizer is None:
                token_list: list[int] = dataset_name[0].tolist()
                token_str_list: list[str] = [str(i) for i in token_list]
                dataset_name_str = "_".join(token_str_list)
            else:
                dataset_name_str = self.tokenizer.decode(
                    dataset_name[0], skip_special_tokens=True
                ).replace(" ", "")
            output_dir = self.output_dir / dataset_name_str
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.output_dir

        if names is None:
            return [self.imgid_to_binpath(i, output_dir) for i in img_ids]

        if not self.use_parent_as_name:
            return [output_dir / Path(names[i]).name for i in img_ids]

        return [
            output_dir / f"{Path(names[i]).parent.name}-{Path(names[i]).name}"
            for i in img_ids
        ]

    def compute_metrics(
        self,
        res: EvalPredictionFlexible,
        compute_result: bool,  # noqa: ARG002
    ) -> dict[str, float]:
        #  When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result` boolean argument which will be triggered after the last batch of the eval set to signal that the summary statistics

        # get local rank and determine if this process should save files
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            should_save = rank == 0
        else:
            rank = 0
            should_save = True

        # Only rank 0 saves files
        result = {}
        if should_save:
            (
                caption_tokens,
                text_ids,
                img_ids,
                text_feat,
                img_feat,
                dataset_name_tokens,
                gen_txt_feat,
            ) = CaptionCache.expand_eval_prediction(res=res)
            result = self._save_captions_and_features(
                caption_tokens,
                text_ids,
                img_ids,
                text_feat,
                img_feat,
                dataset_name_tokens,
                gen_txt_feat,
            )

        # Add barrier after all file operations are complete
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return result

    def _save_captions_and_features(
        self,
        caption_tokens: torch.Tensor,
        text_ids: list[int],
        img_ids: list[int],
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        dataset_name_tokens: torch.Tensor,
        gen_txt_feat: torch.Tensor,
    ) -> dict[str, float]:
        # generated captions
        cache_paths = self.imgid_to_imgpath(
            img_ids=img_ids, dataset_name=dataset_name_tokens
        )

        if gen_txt_feat.shape[0] > 0:
            for path, tokens, gen_feat in zip(
                cache_paths, caption_tokens, gen_txt_feat, strict=True
            ):
                if self.save_features:
                    save_tensor(
                        tensor=gen_feat, path=path.with_suffix(".gen_feat_bin")
                    )

                if self.tokenizer is None:
                    save_tensor(tensor=tokens, path=path)

        if self.tokenizer is not None:
            name_paths = self.imgid_to_imgpath(
                img_ids=img_ids,
                dataset_name=dataset_name_tokens,
                names=self.img_names,
            )
            for path, tokens in zip(name_paths, caption_tokens, strict=True):
                gen_captions: list[str] = self.tokenizer.batch_decode(
                    tokens, skip_special_tokens=True
                )
                with Path(path).with_suffix(".txt").open("w") as f:
                    f.writelines(
                        caption.replace("\n", "") + "\n" for caption in gen_captions
                    )

        # image features
        if not self.save_features:
            return {}

        img_paths = [
            Path(p).parent / ("img_feat_" + Path(p).name) for p in cache_paths
        ]
        for path, feat in zip(img_paths, img_feat, strict=True):
            save_tensor(tensor=feat, path=path)

        # text features
        _text_paths = self.imgid_to_imgpath(
            img_ids=text_ids, dataset_name=dataset_name_tokens
        )
        text_paths = [
            Path(p).parent / ("text_feat_" + Path(p).name) for p in _text_paths
        ]
        for path, feat in zip(text_paths, text_feat, strict=True):
            save_tensor(tensor=feat, path=path)

        return {}
