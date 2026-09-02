from dataclasses import dataclass

import torch
from torch.nn import KLDivLoss

from capcl import registry
from capcl.metrics.metrics import MetricCalculator

CORRECT_PRECISION_THRESHOLD = 0.5


@dataclass
class DistillInput:
    t2i_sim: torch.Tensor
    i2t_sim: torch.Tensor
    prev_t2i_sim: torch.Tensor
    prev_i2t_sim: torch.Tensor
    t2i_label: torch.Tensor
    i2t_label: torch.Tensor
    img_feats: torch.Tensor
    txt_feats: torch.Tensor
    prev_img_feats: torch.Tensor
    prev_txt_feats: torch.Tensor


class SimDistillLoss:
    def __init__(
        self, use_img_intra_loss: bool = True, apply_softmax: bool = True
    ) -> None:
        self.kl_loss_fn = KLDivLoss(reduction="batchmean")
        self.mse_loss_fn = torch.nn.MSELoss(reduction="sum")
        self.use_img_intra_loss = use_img_intra_loss  # for c2mr loss
        self.n_sum_loss = 4.0 if use_img_intra_loss else 3.0  # for c2mr loss
        self.apply_softmax = apply_softmax

    def compute(
        self,
        distill_input: DistillInput,
    ) -> torch.Tensor:
        msg = "compute method is not implemented."
        raise NotImplementedError(msg)

    def get_correct_retrieval(
        self, t2i_sim: torch.Tensor, t2i_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t2i_n_positives = (t2i_label.sum(dim=1).long() - 1).clamp(min=0)
        i2t_n_positives = (t2i_label.sum(dim=0).long() - 1).clamp(min=0)

        t2i_sorted = MetricCalculator.calc_sorted_labels(t2i_sim, t2i_label)
        i2t_sorted = MetricCalculator.calc_sorted_labels(t2i_sim.T, t2i_label.T)

        t2i_p = (
            t2i_sorted.cumsum(1)[torch.arange(t2i_label.size(0)), t2i_n_positives]
            / t2i_n_positives
        )
        i2t_p = (
            i2t_sorted.cumsum(1)[torch.arange(t2i_label.size(1)), i2t_n_positives]
            / i2t_n_positives
        )

        t2i_target_ids = t2i_p > CORRECT_PRECISION_THRESHOLD
        i2t_target_ids = i2t_p > CORRECT_PRECISION_THRESHOLD

        return t2i_target_ids, i2t_target_ids

    def _sim_preprocess(self, sim: torch.Tensor) -> torch.Tensor:
        if not self.apply_softmax:
            return torch.nn.functional.sigmoid(sim)

        return sim.softmax(dim=-1)


@registry.register_sim_distill_loss("kl")
class KLDistillLoss(SimDistillLoss):
    def compute(self, distill_input: DistillInput) -> torch.Tensor:
        prev_t2i_processed = self._sim_preprocess(distill_input.prev_t2i_sim)
        prev_i2t_processed = self._sim_preprocess(distill_input.prev_i2t_sim)

        t2i_processed = self._sim_preprocess(distill_input.t2i_sim)
        i2t_processed = self._sim_preprocess(distill_input.i2t_sim)

        # compute the loss
        t2i_loss: torch.Tensor = self.kl_loss_fn(
            t2i_processed.log(),
            prev_t2i_processed.detach(),
        )

        i2t_loss: torch.Tensor = self.kl_loss_fn(
            i2t_processed.log(),
            prev_i2t_processed.detach(),
        )

        return t2i_loss + i2t_loss


@registry.register_sim_distill_loss("mse")
class MSEDistillLoss(SimDistillLoss):
    def compute(self, distill_input: DistillInput) -> torch.Tensor:
        prev_t2i_processed = self._sim_preprocess(distill_input.prev_t2i_sim)
        prev_i2t_processed = self._sim_preprocess(distill_input.prev_i2t_sim)

        t2i_processed = self._sim_preprocess(distill_input.t2i_sim)
        i2t_processed = self._sim_preprocess(distill_input.i2t_sim)

        # compute the loss
        t2i_batch_size = t2i_processed.size(0)
        t2i_loss: torch.Tensor = (
            self.mse_loss_fn(
                t2i_processed,
                prev_t2i_processed.detach(),
            )
            / t2i_batch_size
        )

        i2t_batch_size = i2t_processed.size(0)
        i2t_loss: torch.Tensor = (
            self.mse_loss_fn(
                i2t_processed,
                prev_i2t_processed.detach(),
            )
            / i2t_batch_size
        )

        return t2i_loss + i2t_loss
