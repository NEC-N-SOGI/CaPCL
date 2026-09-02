import torch

from capcl.models.retriever import Retriever


def soft_nn_loss(
    cos_sim_mat: torch.Tensor,
    pair_onehot: torch.Tensor,
    use_tanh_scale: bool = True,
) -> torch.Tensor:
    # https://arxiv.org/abs/2501.17683
    if use_tanh_scale:
        logit_exp = (1 + cos_sim_mat) / (1 - cos_sim_mat + 1e-6)
    else:
        logit_exp = cos_sim_mat.exp()

    sum_exp_dist = torch.sum(logit_exp, dim=1, keepdim=True)
    sum_positive = torch.sum(logit_exp * pair_onehot, dim=1, keepdim=True)

    _loss = sum_positive / sum_exp_dist

    return -torch.log(_loss).mean()


class InfoNCECrossModalLoss:
    def __init__(
        self, temperature: float = 1.0, use_tanh_scale: bool = False
    ) -> None:
        self.temperature = temperature
        self.use_tanh_scale = use_tanh_scale

    def scale_sim(self, sim: torch.Tensor) -> torch.Tensor:
        return sim / self.temperature

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

        loss = (
            soft_nn_loss(t2i_sim_scaled, t2i_label, self.use_tanh_scale)
            + soft_nn_loss(i2t_sim_scaled, i2t_label, self.use_tanh_scale)
        ) / 2

        if return_sims:
            return t2i_sim_scaled, i2t_sim_scaled, t2i_label, i2t_label, loss
        return loss
