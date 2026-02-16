from __future__ import annotations

import torch


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


def _last_token(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).long().clamp_min(1) - 1
    batch_idx = torch.arange(h.size(0), device=h.device)
    return h[batch_idx, lengths, :]


def pool_hidden(
    h: torch.Tensor,
    mask: torch.Tensor,
    strategy: str = "mean",
    step_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool token representations into one trajectory embedding.

    - mean: masked mean over valid tokens
    - last: last non-pad token
    - last_k_step: masked mean on provided step_mask (fallback to last)
    """

    if strategy == "mean":
        return _masked_mean(h, mask)

    if strategy == "last":
        return _last_token(h, mask)

    if strategy == "last_k_step":
        if step_mask is None:
            return _last_token(h, mask)
        combined = (mask > 0) & (step_mask > 0)
        combined_f = combined.float()
        has_region = combined.any(dim=1)
        pooled_step = _masked_mean(h, combined_f)
        pooled_last = _last_token(h, mask)
        selector = has_region.unsqueeze(-1)
        return torch.where(selector, pooled_step, pooled_last)

    raise ValueError(f"Unknown pooling strategy: {strategy}")
