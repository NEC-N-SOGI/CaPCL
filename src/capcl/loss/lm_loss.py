import torch

PAD_TOKEN_ID = -100


class NextTokenPredictionLoss:
    def __init__(self, cap_reduction: str = "mean") -> None:
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=PAD_TOKEN_ID, reduction="none"
        )
        self.cap_reduction = cap_reduction

    def _reduce_caps(
        self, bs_cap_loss_lm: torch.Tensor, n_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Reduce the caption loss tensor.

        Args:
            bs_cap_loss_lm (torch.Tensor): The batch size caption loss tensor. shape = (batch_size, n_caps)
            n_tokens (torch.Tensor): The number of tokens in each caption. shape = (batch_size, n_caps)

        Returns:
            torch.Tensor: loss for each sample. shape = (batch_size,)
        """
        if self.cap_reduction == "max":
            return bs_cap_loss_lm.max(-1).values

        if self.cap_reduction == "min":
            bs_cap_loss_lm[n_tokens == 0] = 10000.0
            return bs_cap_loss_lm.min(-1).values

        seq_num = (n_tokens > 0).sum(-1).clamp(min=1.0)
        bs_loss_lm: torch.Tensor = bs_cap_loss_lm.sum(-1) / seq_num
        return bs_loss_lm

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the next token prediction loss.

        Args:
            logits (torch.Tensor): shape (batch_size, n_caps, n_tokens, vocab_size)
            labels (torch.Tensor): shape (batch_size, n_caps, n_tokens)

        Returns:
            torch.Tensor: scalar loss value
        """
        n_bs = logits.size(0)
        n_caps = logits.size(1)
        vocab_size = logits.size(-1)

        # Shift for next token prediction
        shifted_logits = logits[
            ..., :-1, :
        ].contiguous()  # [bs, n_caps, seq-1, vocab]
        shifted_labels = labels[..., 1:].contiguous()  # [bs, n_caps, seq-1]

        # Count valid tokens per caption
        n_tokens = (
            (shifted_labels != PAD_TOKEN_ID).sum(-1).clamp(min=1)
        )  # [bs, n_caps]

        # Process caption by caption to save memory
        bs_cap_loss_lm = torch.zeros(
            n_bs, n_caps, device=logits.device, dtype=logits.dtype
        )

        for cap_idx in range(n_caps):
            # Extract single caption: [bs, seq-1, vocab] and [bs, seq-1]
            cap_logits = shifted_logits[:, cap_idx, :, :]  # [bs, seq-1, vocab]
            cap_labels = shifted_labels[:, cap_idx, :]  # [bs, seq-1]

            # Flatten batch and sequence: [bs*(seq-1), vocab] and [bs*(seq-1)]
            cap_logits_flat = cap_logits.reshape(-1, vocab_size)
            cap_labels_flat = cap_labels.reshape(-1)

            # Compute loss: [bs*(seq-1)]
            loss_per_token = self.loss_fct(cap_logits_flat, cap_labels_flat)

            # Reshape and sum: [bs, seq-1] -> [bs]
            loss_per_sample = loss_per_token.view(n_bs, -1).sum(-1)

            # Average by valid tokens
            bs_cap_loss_lm[:, cap_idx] = loss_per_sample / n_tokens[:, cap_idx]

        # mean along with n_caps, then the batch size
        return self._reduce_caps(bs_cap_loss_lm, n_tokens).mean()
