from pathlib import Path

import numpy as np
import torch
from transformers import EvalPrediction

from capcl.io.tensor_rw import save_tensor
from capcl.metrics.eval_metrics import EvalMetrics
from capcl.models.retriever import Retriever


def calc_map_at_k(
    pred_label: torch.Tensor, top_k: int = 20
) -> tuple[torch.Tensor, float]:
    if pred_label.ndim == 1:
        pred_label = pred_label.unsqueeze(0)
    n_positive = pred_label.sum(1)

    top_k = min(top_k, pred_label.shape[1])
    # elementwise min(n_positive, top_k)
    n_k = torch.minimum(n_positive, torch.tensor(top_k, dtype=torch.float32))

    pred_at_k = pred_label[:, :top_k]

    ap = (pred_at_k.cumsum(1) / (torch.arange(top_k) + 1) * pred_at_k).sum(1) / n_k
    ap[ap.isnan()] = 0

    mean_ap = ap.mean()

    return ap, mean_ap.item()


def calc_precision_at_k(
    pred_label: torch.Tensor, top_k: int = 20
) -> tuple[torch.Tensor, float]:
    if pred_label.ndim == 1:
        pred_label = pred_label.unsqueeze(0)
    n_positive = pred_label.sum(1)

    top_k = min(top_k, pred_label.shape[1])
    # elementwise min(n_positive, top_k)
    n_k = torch.minimum(n_positive, torch.tensor(top_k, dtype=torch.float32))

    pred_at_k = pred_label[:, :top_k]

    precision = pred_at_k.cumsum(1) / (torch.arange(top_k) + 1)

    # extract precision at n_k, i.e., precision[i, n_k[i]-1]
    precision = precision[torch.arange(precision.shape[0]), n_k.long() - 1]
    mean_precision = precision.mean()

    return precision, mean_precision.item()


def calc_r_at_k(
    pred_label: torch.Tensor,
) -> tuple[float, float, float]:
    if pred_label.ndim == 1:
        pred_label = pred_label.unsqueeze(0)

    pred_label = pred_label.to(torch.float32)
    r1 = pred_label[:, :1].max(-1)[0].mean().item()
    r5 = pred_label[:, :5].max(-1)[0].mean().item()
    r10 = pred_label[:, :10].max(-1)[0].mean().item()

    return r1, r5, r10


def calc_mrr(pred_label: torch.Tensor) -> float:
    if pred_label.ndim == 1:
        pred_label = pred_label.unsqueeze(0)

    pred_label = pred_label.to(torch.float32)
    mrr: torch.Tensor = (1 / (pred_label.argmax(-1) + 1)).mean()

    return float(mrr.item())


class MetricCalculator:
    def __init__(self, top_k: int = 20, output_dir: Path | None = None) -> None:
        self.top_k = top_k
        self.output_dir = output_dir

    @staticmethod
    def expand_eval_prediction(
        res: EvalPrediction,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # res.label_ids = (label_t2i, text_ids, image_ids, dataset_names) defined by transformers.TrainingArguments.label_names.
        if len(res.label_ids) < 3:  # noqa: PLR2004
            msg = f"label_ids must have at least 3 elements (label_t2i, text_ids, image_ids), but got {len(res.label_ids)}"
            raise ValueError(msg)
        if not (
            isinstance(res.label_ids, tuple)
            and isinstance(res.label_ids[0], np.ndarray)
        ):
            msg = f"label_ids must be a tuple of np.ndarray, but got {type(res.label_ids)}"
            raise TypeError(msg)

        label_ids: tuple[np.ndarray, np.ndarray, np.ndarray]
        label_ids = (res.label_ids[0], res.label_ids[1], res.label_ids[2])  # ty: ignore[index-out-of-bounds]

        label_t2i_np, txt_ids_np, img_ids_np = label_ids
        label_t2i, txt_ids, img_ids = (
            torch.tensor(label_t2i_np, device="cpu"),
            torch.tensor(txt_ids_np, device="cpu"),
            torch.tensor(img_ids_np, device="cpu"),
        )

        # res.predictions = (txt_feat, img_feat, ...) defined by capcl.models.ModelOutput.
        # Note that, "loss" is dropped by transformers.trainer.Trainer.prediction_step. (see transformers.trainer.py line 4473)
        if len(res.predictions) < 4:  # noqa: PLR2004
            msg = f"predictions must have at least 4 elements (txt_feat, img_feat, ...), but got {len(res.predictions)}"
            raise ValueError(msg)
        if not (
            isinstance(res.predictions, tuple)
            and isinstance(res.predictions[0], np.ndarray)
        ):
            msg = f"predictions must be a tuple of np.ndarray, but got {type(res.predictions)}"
            raise TypeError(msg)

        txt_feat_np, img_feat_np = res.predictions[0], res.predictions[1]  # ty: ignore[index-out-of-bounds]

        txt_feat, img_feat = (
            torch.tensor(txt_feat_np, device="cpu"),
            torch.tensor(img_feat_np, device="cpu"),
        )

        dataset_names_np: np.ndarray = res.predictions[3]  # ty: ignore[index-out-of-bounds]
        dataset_names = torch.tensor(dataset_names_np, device="cpu")

        return txt_feat, img_feat, label_t2i, txt_ids, img_ids, dataset_names

    @staticmethod
    def calc_sorted_labels(sim: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        _, sorted_idx = sim.sort(dim=-1, descending=True)

        return label[torch.arange(label.size(0)).unsqueeze(1), sorted_idx]

    def compute_metrics(self, res: EvalPrediction) -> dict[str, float | int]:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        txt_feat, img_feat, label_t2i, txt_ids, img_ids, dataset_names = (
            MetricCalculator.expand_eval_prediction(res=res)
        )

        t2i_label, i2t_label, sim_t2i, sim_i2t = Retriever.run(
            txt_feat=txt_feat,
            img_feat=img_feat,
            txt_ids=txt_ids,
            img_ids=img_ids,
            label_t2i=label_t2i,
        )
        n_imgs, n_txts = sim_i2t.shape

        t2i_sorted_labels = MetricCalculator.calc_sorted_labels(sim_t2i, t2i_label)
        i2t_sorted_labels = MetricCalculator.calc_sorted_labels(sim_i2t, i2t_label)

        # metrics
        t2i_r1, t2i_r5, t2i_r10 = calc_r_at_k(t2i_sorted_labels)
        t2i_mr = calc_mrr(t2i_sorted_labels)

        t2i_map = {}
        i2t_map = {}
        for k in [10, 50, 100, 1000]:
            _, t2i_map[f"t2i_map_{k}"] = calc_map_at_k(t2i_sorted_labels, top_k=k)
            _, i2t_map[f"i2t_map_{k}"] = calc_map_at_k(i2t_sorted_labels, top_k=k)

        t2i_pk = {}
        i2t_pk = {}
        for k in [10, 50, 100, 1000]:
            _, t2i_pk[f"t2i_pk_{k}"] = calc_precision_at_k(
                t2i_sorted_labels, top_k=k
            )
            _, i2t_pk[f"i2t_pk_{k}"] = calc_precision_at_k(
                i2t_sorted_labels, top_k=k
            )

        i2t_r1, i2t_r5, i2t_r10 = calc_r_at_k(i2t_sorted_labels)
        i2t_mr = calc_mrr(i2t_sorted_labels)

        if self.output_dir is not None:
            dataset_name = [str(i) for i in dataset_names[0].tolist() if i > 0]
            prefix = "_".join(dataset_name)

            if rank == 0:
                output_path = self.output_dir / f"{prefix}_sim_t2i.bin"
                save_tensor(sim_t2i, output_path)

                output_path = self.output_dir / f"{prefix}_t2i_label.bin"
                save_tensor(t2i_label, output_path)

        metrics: dict[str, float] = EvalMetrics(
            n_imgs=n_imgs,
            n_txts=n_txts,
            t2i_r1=t2i_r1,
            t2i_r5=t2i_r5,
            t2i_r10=t2i_r10,
            t2i_mrr=t2i_mr,
            i2t_r1=i2t_r1,
            i2t_r5=i2t_r5,
            i2t_r10=i2t_r10,
            i2t_mrr=i2t_mr,
            **t2i_map,
            **i2t_map,
            **t2i_pk,
            **i2t_pk,
        ).model_dump()

        if not all(isinstance(v, (float, int)) for v in metrics.values()):
            msg = (
                "All values in metrics must be float, but got "
                f"{[(k, type(v)) for k, v in metrics.items()]}"
            )
            raise ValueError(msg)

        return metrics
