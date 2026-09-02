import torch

from capcl.models.modules.normalize import l2_normalization


class Retriever:
    @staticmethod
    def remove_duplicate(
        ids: torch.Tensor,
        data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        passed_id: set[int] = set()
        n_data = len(set(ids.cpu().numpy()) - {-1})

        sort_idx = torch.argsort(ids, descending=False)

        # remove duplicate
        sort_idx_wo_dup = torch.zeros(n_data, dtype=torch.int, device=ids.device)

        idx_counter = 0
        for i in sort_idx:
            id_int = int(ids[i].item())  # Avoid CPU transfer
            if id_int in passed_id or id_int < 0:
                continue

            sort_idx_wo_dup[idx_counter] = i
            passed_id.add(id_int)
            idx_counter += 1

        return data[sort_idx_wo_dup].contiguous(), sort_idx_wo_dup

    @staticmethod
    def cos_sim_max(img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        if img_feat.ndim == 3:  # noqa: PLR2004 (3D tensor)
            # txt_feat shape = [n_txt, dim] --> [n_txt, 1, 1, dim]
            # img_feat shape = [n_img, 32, dim] --> [n_img, dim, 32]
            # sim_t2q shape = [n_txt, n_img, 1, 32]
            sim_t2q = torch.matmul(
                txt_feat.unsqueeze(1).unsqueeze(1), img_feat.permute(0, 2, 1)
            )
            sim_t2i = sim_t2q.max(dim=-1)[0].squeeze(dim=-1)
        else:
            sim_t2i = torch.matmul(txt_feat, img_feat.T)

        return sim_t2i

    @staticmethod
    def calc_similarity(
        txt_feat: torch.Tensor, img_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        txt_feat = l2_normalization(txt_feat)
        img_feat = l2_normalization(img_feat)

        # sims
        if txt_feat.ndim == 1:
            txt_feat = txt_feat.unsqueeze(0)

        if img_feat.ndim == 1:
            img_feat = img_feat.unsqueeze(0)

        n_bs = 100
        device = txt_feat.device

        # Pre-allocate on GPU to avoid repeated allocations
        sim_t2i = torch.zeros(
            (txt_feat.shape[0], img_feat.shape[0]),
            device=device,
            dtype=txt_feat.dtype,
        )

        for i in range(0, txt_feat.shape[0], n_bs):
            for j in range(0, img_feat.shape[0], n_bs):
                # Compute in-place to save memory
                sim_t2i[i : i + n_bs, j : j + n_bs] = Retriever.cos_sim_max(
                    img_feat[j : j + n_bs],
                    txt_feat[i : i + n_bs],
                )

        if sim_t2i.dim() == 1:
            expand_dim = 0 if txt_feat.shape[0] == 1 else 1
            sim_t2i = sim_t2i.unsqueeze(expand_dim)

        sim_i2t = sim_t2i.T.contiguous()

        return sim_t2i, sim_i2t

    @staticmethod
    def orgnize_prediction(
        txt_feat_dup: torch.Tensor,
        img_feat_dup: torch.Tensor,
        label_t2i_dup: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        txt_feat, txt_sort_idx = Retriever.remove_duplicate(txt_ids, txt_feat_dup)
        label_t2i_w_img_dup = label_t2i_dup[txt_sort_idx]

        img_feat, img_sort_idx = Retriever.remove_duplicate(img_ids, img_feat_dup)
        label_t2i = label_t2i_w_img_dup[:, img_ids[img_sort_idx]]

        if txt_feat.dim() > 2:  # noqa: PLR2004
            # [bs, n_tokens, dim] -> [bs, dim]
            txt_feat = txt_feat[:, 0]

        return label_t2i, txt_feat, img_feat

    @staticmethod
    def run(
        txt_feat: torch.Tensor,
        img_feat: torch.Tensor,
        label_t2i: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t2i_label_sorted_wo_dup, txt_feat_sorted_wo_dup, img_feat_sorted_wo_dup = (
            Retriever.orgnize_prediction(
                txt_feat, img_feat, label_t2i, txt_ids, img_ids
            )
        )

        sim_t2i, sim_i2t = Retriever.calc_similarity(
            txt_feat_sorted_wo_dup, img_feat_sorted_wo_dup
        )

        if sim_i2t.dim() == 1:
            sim_i2t = sim_i2t.unsqueeze(1)

        if sim_t2i.dim() == 1:
            sim_t2i = sim_t2i.unsqueeze(0)

        return t2i_label_sorted_wo_dup, t2i_label_sorted_wo_dup.T, sim_t2i, sim_i2t
