"""Action-conditioned visual-token sensitivity scoring utilities.

These helpers are intentionally parameter-free so they can be enabled at
inference without retraining.  They score visual tokens by combining generic
activation saliency with AEO delta sensitivity and optional action-context
alignment, then keep the most sensitive tokens while preserving sequence order.
"""

from __future__ import annotations

import torch


def _normalize_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.float().masked_fill(~mask.bool(), 0.0)
    denom = scores.amax(dim=1, keepdim=True).clamp_min(1e-6)
    return scores / denom


def score_action_sensitive_tokens(
    token_embs: torch.Tensor,
    token_masks: torch.Tensor,
    *,
    action_context: torch.Tensor | None = None,
    delta_embs: torch.Tensor | None = None,
    norm_weight: float = 0.25,
    delta_weight: float = 0.65,
    action_weight: float = 0.10,
) -> torch.Tensor:
    """Score how sensitive each visual token is for the current action step.

    Args:
        token_embs: Visual token embeddings with shape ``[B, N, D]``.
        token_masks: Valid-token mask with shape ``[B, N]``.
        action_context: Optional current action residual / action-left vector.
            If provided, the vector is zero-padded or truncated to the embedding
            dimension and used as a weak alignment signal.
        delta_embs: Optional AEO predictor delta embeddings with shape
            ``[B, N, D]``.  Per-token delta norm is the main sensitivity signal
            because it measures how much AEO changes each visual token.
        norm_weight: Weight for generic token activation saliency.
        delta_weight: Weight for AEO delta saliency.
        action_weight: Weight for action-context alignment saliency.

    Returns:
        A ``[B, N]`` score tensor; invalid tokens are assigned ``-inf``.
    """
    if token_embs.ndim != 3:
        raise ValueError(f"token_embs must have shape [B, N, D], got {tuple(token_embs.shape)}")
    if token_masks.ndim != 2:
        raise ValueError(f"token_masks must have shape [B, N], got {tuple(token_masks.shape)}")
    if token_embs.shape[:2] != token_masks.shape:
        raise ValueError(f"token_embs and token_masks disagree: {tuple(token_embs.shape[:2])} vs {tuple(token_masks.shape)}")

    valid_mask = token_masks.bool()
    activation_scores = _normalize_scores(torch.linalg.vector_norm(token_embs.float(), ord=2, dim=-1), valid_mask)
    total_scores = norm_weight * activation_scores

    if delta_embs is not None:
        if delta_embs.shape != token_embs.shape:
            raise ValueError(f"delta_embs must match token_embs, got {tuple(delta_embs.shape)} vs {tuple(token_embs.shape)}")
        delta_scores = _normalize_scores(torch.linalg.vector_norm(delta_embs.float(), ord=2, dim=-1), valid_mask)
        total_scores = total_scores + delta_weight * delta_scores

    if action_context is not None:
        action_tensor = torch.as_tensor(action_context, dtype=token_embs.dtype, device=token_embs.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor[None, :].expand(token_embs.shape[0], -1)
        action_tensor = action_tensor.reshape(token_embs.shape[0], -1)
        hidden_dim = token_embs.shape[-1]
        if action_tensor.shape[-1] < hidden_dim:
            action_tensor = torch.nn.functional.pad(action_tensor, (0, hidden_dim - action_tensor.shape[-1]))
        else:
            action_tensor = action_tensor[:, :hidden_dim]
        action_tensor = torch.nn.functional.normalize(action_tensor.float(), dim=-1)
        token_dirs = torch.nn.functional.normalize(token_embs.float(), dim=-1)
        alignment_scores = (token_dirs * action_tensor[:, None, :]).sum(dim=-1).abs()
        alignment_scores = _normalize_scores(alignment_scores, valid_mask)
        total_scores = total_scores + action_weight * alignment_scores

    return total_scores.masked_fill(~valid_mask, torch.finfo(total_scores.dtype).min)


def keep_top_sensitive_tokens(
    token_embs: torch.Tensor,
    token_masks: torch.Tensor,
    keep_ratio: float,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep top-scoring visual tokens and preserve their original order."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if scores.shape != token_masks.shape:
        raise ValueError(f"scores and token_masks disagree: {tuple(scores.shape)} vs {tuple(token_masks.shape)}")

    batch_size, num_tokens, hidden_dim = token_embs.shape
    keep_count = max(1, int(torch.ceil(torch.tensor(num_tokens * keep_ratio)).item()))
    if keep_count >= num_tokens:
        indices = torch.arange(num_tokens, device=token_embs.device).expand(batch_size, num_tokens)
        return token_embs, token_masks, indices

    indices = torch.topk(scores, k=keep_count, dim=1, largest=True, sorted=False).indices
    indices = torch.sort(indices, dim=1).values
    gather_indices = indices.unsqueeze(-1).expand(batch_size, keep_count, hidden_dim)
    kept_embs = torch.gather(token_embs, dim=1, index=gather_indices)
    kept_masks = torch.gather(token_masks.bool(), dim=1, index=indices)
    return kept_embs, kept_masks, indices
