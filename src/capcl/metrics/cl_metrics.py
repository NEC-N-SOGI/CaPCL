from pathlib import Path
from typing import Any

import torch
from natsort import natsorted

from capcl.dataset.dataset_dict import DatasetDictModel


class CLMetricsCalculator:
    """Base class for calculating CL metrics."""

    def __init__(self, dataset_path: Path) -> None:
        self.metrics: dict[str, float] = {}
        self.dataset_path = dataset_path

        dataset_model = DatasetDictModel.from_json(dataset_path)

        self.train_names = [k for k in dataset_model.dataset if "train" in k]
        self.test_names = [k for k in dataset_model.dataset if "test" in k]

        self.n_train_tasks = len(self.train_names)
        self.n_test_tasks = len(self.test_names)

        # 0: no pretrained tasks. just use the first column as the first task.
        # >1: we have some pretrained tasks, we use the first column as the pretrained results.
        # Note: all data# in a dataset json file must have the "test" split
        # Note: No train split means it is a pretrained task.
        self.n_pretrained_tasks = max(1, self.n_test_tasks - self.n_train_tasks)

        self.i2t_metrics: torch.Tensor | None = None
        self.t2i_metrics: torch.Tensor | None = None

        # Initialize i2t and t2i metrics based on the dataset model
        self.target_i2t = natsorted(
            [
                f"{k}_i2t_{v}"
                for k, v in dataset_model.target_metric.items()
                if "test" in k
            ]
        )

        self.target_t2i = natsorted(
            [
                f"{k}_t2i_{v}"
                for k, v in dataset_model.target_metric.items()
                if "test" in k
            ]
        )

    def search_endwiths(
        self, target_dict: dict[str, Any], endwith: str
    ) -> list[Any]:
        """Search for keys in the dictionary that end with a specific suffix."""
        # Huggingface transformers add prefixes (train/eval/val) to the metric names,
        # so we need to search for the suffix.
        return [v for k, v in target_dict.items() if k.endswith(endwith)]

    def concat_metrics(
        self, metrics: dict[str, float], target: list, old: torch.Tensor | None
    ) -> torch.Tensor:
        """Concatenate new metrics to the existing metrics tensor."""
        new_metrics = torch.zeros(self.n_test_tasks, dtype=torch.float32)

        for i, target_name in enumerate(target):
            metric = self.search_endwiths(metrics, target_name)[0]
            new_metrics[i] = metric

        if old is None:
            return new_metrics.unsqueeze(1)

        return torch.cat([old, new_metrics.unsqueeze(1)], dim=1)

    def add_metrics(self, metrics: dict[str, float]) -> None:
        """Register new metrics."""
        self.i2t_metrics = self.concat_metrics(
            metrics, self.target_i2t, self.i2t_metrics
        )
        self.t2i_metrics = self.concat_metrics(
            metrics, self.target_t2i, self.t2i_metrics
        )

    def _harmonic_mean(self, x: float, y: float, eps: float = 1e-8) -> float:
        """Calculate the harmonic mean of two values."""
        return 2 / (1 / x + 1 / y + eps)

    def calc_each_metric(
        self, accs: torch.Tensor, idx: int = -1
    ) -> tuple[float, float, float, float, float]:
        """Calculate average accuracy, average incremental accuracy, and forgetting measure."""
        n_pretrained_datasets = self.n_pretrained_tasks

        # lower off-diagonal elements are evaluated results before training their tasks.
        learned_acc = torch.triu(accs, diagonal=-(n_pretrained_datasets - 1))

        # n learned task on each time
        n_learned_task = torch.arange(
            1,
            learned_acc.size(1) + 1,
            dtype=accs.dtype,
            device=learned_acc.device,
        ) + (n_pretrained_datasets - 1)

        average_acc = learned_acc.sum(dim=0) / n_learned_task
        average_incremental_acc = average_acc.cumsum(dim=0) / torch.arange(
            1, len(average_acc) + 1
        )

        # forgetting measure
        # get the best accuracy for each task at each time step
        best_acc = torch.stack(
            [learned_acc[:, :i].max(1).values for i in range(1, learned_acc.size(1))]
        ).T

        # calculate the difference between the best accuracy and the current accuracy
        best_diff = torch.triu(
            best_acc - learned_acc[:, 1:], diagonal=-(n_pretrained_datasets - 1)
        )

        fm = best_diff.sum(0) / n_learned_task[:-1]
        fm = torch.cat([torch.tensor([0.0], device=fm.device), fm])

        return (
            average_acc[idx].item(),
            average_incremental_acc[idx].item(),
            fm[idx].item(),
            self._harmonic_mean(average_acc[idx].item(), (1 - fm[idx].item())),
            self._harmonic_mean(
                average_incremental_acc[idx].item(), (1 - fm[idx].item())
            ),
        )

    def calculate_metrics(self, new_metrics: dict[str, float]) -> dict[str, float]:
        """Calculate and return the CL metrics with new metrics."""
        self.add_metrics(new_metrics)

        if self.i2t_metrics is None or self.t2i_metrics is None:
            msg = (
                "Metrics for i2t or t2i are not initialized. "
                "Please ensure that metrics are added before calculating."
            )
            raise ValueError(msg)

        i2t_aa, i2t_aia, i2t_fm, i2t_aa_hm, i2t_aia_hm = self.calc_each_metric(
            self.i2t_metrics
        )
        t2i_aa, t2i_aia, t2i_fm, t2i_aa_hm, t2i_aia_hm = self.calc_each_metric(
            self.t2i_metrics
        )

        return {
            "i2t_aa": i2t_aa,
            "i2t_aia": i2t_aia,
            "i2t_fm": i2t_fm,
            "t2i_aa": t2i_aa,
            "t2i_aia": t2i_aia,
            "t2i_fm": t2i_fm,
            "t2i_aa_hm": t2i_aa_hm,
            "t2i_aia_hm": t2i_aia_hm,
            "i2t_aa_hm": i2t_aa_hm,
            "i2t_aia_hm": i2t_aia_hm,
        }

    def make_table(self) -> dict[str, list[Any]]:
        if self.t2i_metrics is None or self.i2t_metrics is None:
            msg = (
                "Metrics for i2t or t2i are not initialized. "
                "Please ensure that metrics are added before making the table."
            )
            raise ValueError(msg)

        table: dict[str, list[Any]] = {}
        table["task_id"] = [f"{i + 1:02d}" for i in range(self.t2i_metrics.size(1))]

        for i, target_name in enumerate(self.target_t2i):
            table[target_name] = self.t2i_metrics[i, :].tolist()

        for i, target_name in enumerate(self.target_i2t):
            table[target_name] = self.i2t_metrics[i, :].tolist()

        t2i_aa = []
        t2i_aia = []
        t2i_fm = []
        t2i_aa_hm = []
        t2i_aia_hm = []
        i2t_aa = []
        i2t_aia = []
        i2t_fm = []
        i2t_aa_hm = []
        i2t_aia_hm = []

        for i in range(self.t2i_metrics.size(1)):
            (
                i2t_aa_i,
                i2t_aia_i,
                i2t_fm_i,
                i2t_aa_hm_i,
                i2t_aia_hm_i,
            ) = self.calc_each_metric(self.i2t_metrics, idx=i)
            (
                t2i_aa_i,
                t2i_aia_i,
                t2i_fm_i,
                t2i_aa_hm_i,
                t2i_aia_hm_i,
            ) = self.calc_each_metric(self.t2i_metrics, idx=i)

            i2t_aa.append(i2t_aa_i)
            i2t_aia.append(i2t_aia_i)
            i2t_fm.append(i2t_fm_i)
            i2t_aa_hm.append(i2t_aa_hm_i)
            i2t_aia_hm.append(i2t_aia_hm_i)

            t2i_aa.append(t2i_aa_i)
            t2i_aia.append(t2i_aia_i)
            t2i_fm.append(t2i_fm_i)
            t2i_aa_hm.append(t2i_aa_hm_i)
            t2i_aia_hm.append(t2i_aia_hm_i)

        table["i2t_aa"] = i2t_aa
        table["i2t_aia"] = i2t_aia
        table["i2t_fm"] = i2t_fm
        table["t2i_aa"] = t2i_aa
        table["t2i_aia"] = t2i_aia
        table["t2i_fm"] = t2i_fm
        table["t2i_aa_hm"] = t2i_aa_hm
        table["t2i_aia_hm"] = t2i_aia_hm
        table["i2t_aa_hm"] = i2t_aa_hm
        table["i2t_aia_hm"] = i2t_aia_hm

        return table
