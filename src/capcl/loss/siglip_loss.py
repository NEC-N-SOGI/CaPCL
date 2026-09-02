import torch

from capcl.models.retriever import Retriever


def sigmoid_loss(
    cos_sim_mat: torch.Tensor,
    pair_onehot: torch.Tensor,
) -> torch.Tensor:
    m1_diag1 = -torch.ones_like(pair_onehot) + 2 * pair_onehot
    loglik = torch.nn.functional.logsigmoid(m1_diag1 * cos_sim_mat)
    nll = -torch.sum(loglik, dim=-1)
    return nll.mean()


class SigmoidLoss:
    def __init__(self, temperature: tuple[float, float] = (1.0, 1.0)) -> None:
        self.temperature = temperature

    def scale_sim(self, sim: torch.Tensor) -> torch.Tensor:
        return sim * self.temperature[0] + self.temperature[1]

    def __call__(
        self,
        txt_feat: torch.Tensor,
        img_feat: torch.Tensor,
        label_t2i: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        return_sims: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        t2i_label, i2t_label, t2i_sim, i2t_sim = Retriever.run(
            txt_feat=txt_feat,
            img_feat=img_feat,
            label_t2i=label_t2i,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

        t2i_sim_scaled = self.scale_sim(t2i_sim)
        i2t_sim_scaled = self.scale_sim(i2t_sim)

        loss = sigmoid_loss(t2i_sim_scaled, t2i_label)

        if return_sims:
            return (
                t2i_sim_scaled,
                i2t_sim_scaled,
                t2i_label,
                i2t_label,
                loss,
            )
        return loss
