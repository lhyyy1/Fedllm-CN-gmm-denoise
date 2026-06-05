from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fate_llm.algo.fedmkt.utils.vars_define import (
    ALIGNED_OTHER_INDICES,
    ALIGNED_OTHER_LOGITS,
    ALIGNED_OTHER_METRIC,
    METRIC,
    PER_STEP_INDICES,
    PER_STEP_LOGITS,
)

"""Trusted semantic alignment modules for FedMKT-style LLM/SLM co-training.

The code is intentionally independent from a concrete Transformer backbone.  It
works on the top-k logits datasets already produced by FedMKT and adds:
  1. shared-private attention autoencoder blocks;
  2. EMA global context + cross attention decoder;
  3. selective SVD reconstruction for cloud-side teacher logits;
  4. dataset utilities that rewrite FedMKT aligned teacher fields.
"""

@dataclass
class TrustAlignConfig:
    vocab_size: int
    top_k: int = 128
    latent_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 4
    ema_decay: float = 0.95
    svd_k_min: int = 8
    svd_k_max: int = 64
    svd_tau: float = 128.0
    private_reg_lambda: float = 1e-4
    temperature: float = 1.0
    device: str = "cpu"


class PrivateCodec(nn.Module):
    """Client-side private encoder/decoder E_p^k, D_p^k.

    Input/output shape: [B, T, vocab_size].  In practice the helper functions
    below densify FedMKT top-k logits before calling this module.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, shared_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, shared_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )
        self._initial_state = {k: v.detach().clone() for k, v in self.encoder.state_dict().items()}

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(h)

    def decode(self, h_shared: torch.Tensor) -> torch.Tensor:
        return self.decoder(h_shared)

    def regularization_loss(self) -> torch.Tensor:
        loss = None
        for name, value in self.encoder.state_dict().items():
            if not torch.is_floating_point(value):
                continue
            init = self._initial_state[name].to(value.device, value.dtype)
            item = (value - init).pow(2).mean()
            loss = item if loss is None else loss + item
        if loss is None:
            loss = torch.tensor(0.0)
        return loss


class SharedAttentionAutoEncoder(nn.Module):
    """Server-side shared encoder/decoder E_s, D_s with MHA and EMA context."""

    def __init__(self, shared_dim: int, latent_dim: int, num_heads: int = 4, ema_decay: float = 0.95):
        super().__init__()
        self.shared_dim = shared_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.input_proj = nn.Linear(shared_dim, latent_dim)
        self.self_attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.encoder_norm = nn.LayerNorm(latent_dim)
        self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.decoder_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, shared_dim)
        self.register_buffer("z_ema", torch.empty(0), persistent=True)

    def encode(self, h_tilde: torch.Tensor) -> torch.Tensor:
        z0 = self.input_proj(h_tilde)
        attn, _ = self.self_attention(z0, z0, z0, need_weights=False)
        return self.encoder_norm(z0 + attn)

    @torch.no_grad()
    def update_ema(self, z_avg: torch.Tensor) -> None:
        z_mean = z_avg.detach().mean(dim=0, keepdim=True)
        if self.z_ema.numel() == 0 or tuple(self.z_ema.shape) != tuple(z_mean.shape):
            self.z_ema = z_mean.clone()
        else:
            self.z_ema.mul_(self.ema_decay).add_(z_mean, alpha=1.0 - self.ema_decay)

    def decode(self, z_avg: torch.Tensor) -> torch.Tensor:
        if self.z_ema.numel() == 0:
            context = z_avg.detach().mean(dim=0, keepdim=True).expand_as(z_avg)
        else:
            context = self.z_ema.to(z_avg.device, z_avg.dtype).expand_as(z_avg)
        cross, _ = self.cross_attention(z_avg, context, context, need_weights=False)
        z = self.decoder_norm(z_avg + cross)
        return self.output_proj(z)

    def forward(self, h_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h_tilde)
        h_shared = self.decode(z)
        return z, h_shared


class HSPAA(nn.Module):
    """Hierarchical shared-private attention autoencoder wrapper."""

    def __init__(self, cfg: TrustAlignConfig):
        super().__init__()
        self.cfg = cfg
        self.private_codec = PrivateCodec(cfg.vocab_size, cfg.hidden_dim, cfg.hidden_dim)
        self.shared_codec = SharedAttentionAutoEncoder(cfg.hidden_dim, cfg.latent_dim, cfg.num_heads, cfg.ema_decay)
        self.to(torch.device(cfg.device))

    def encode_private(self, dense_logits_or_probs: torch.Tensor) -> torch.Tensor:
        return self.private_codec.encode(dense_logits_or_probs)

    def encode_shared(self, h_tilde: torch.Tensor) -> torch.Tensor:
        return self.shared_codec.encode(h_tilde)

    def decode_shared(self, z_avg: torch.Tensor) -> torch.Tensor:
        return self.shared_codec.decode(z_avg)

    def decode_private(self, h_shared: torch.Tensor) -> torch.Tensor:
        return self.private_codec.decode(h_shared)

    def local_reconstruction_loss(self, dense_target: torch.Tensor) -> torch.Tensor:
        h_tilde = self.encode_private(dense_target)
        z = self.encode_shared(h_tilde)
        h_shared = self.decode_shared(z)
        pred = self.decode_private(h_shared)
        mse = F.mse_loss(pred, dense_target)
        return mse + self.cfg.private_reg_lambda * self.private_codec.regularization_loss()


def _safe_metric(metrics: List[float], i: int, default: float = 0.0) -> float:
    try:
        v = metrics[i]
        return float(v) if v is not None else default
    except Exception:
        return default


def sparse_topk_to_dense(
    logits: List[List[float]],
    indices: List[List[int]],
    vocab_size: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Convert per-token top-k logits to dense probability tensor [T, V]."""
    if len(logits) == 0:
        return torch.zeros(0, vocab_size, device=device)
    dense = torch.zeros(len(logits), vocab_size, dtype=torch.float32, device=device)
    for t, (step_logits, step_indices) in enumerate(zip(logits, indices)):
        if len(step_logits) == 0 or len(step_indices) == 0:
            continue
        values = torch.tensor(step_logits, dtype=torch.float32, device=device)
        ids = torch.tensor(step_indices, dtype=torch.long, device=device)
        probs = torch.softmax(values / max(temperature, 1e-6), dim=-1)
        dense[t].scatter_(0, ids, probs)
    return dense


def dense_to_sparse_topk(dense_logits: torch.Tensor, top_k: int) -> Tuple[List[List[float]], List[List[int]]]:
    if dense_logits.numel() == 0:
        return [], []
    k = min(int(top_k), dense_logits.shape[-1])
    values, ids = torch.topk(dense_logits.detach().float().cpu(), k=k, dim=-1)
    return values.tolist(), ids.tolist()


def dynamic_svd_rank(round_idx: int, total_rounds: int, cfg: TrustAlignConfig, acc: float = 0.0) -> int:
    progress_rank = cfg.svd_k_min + (cfg.svd_k_max - cfg.svd_k_min) * float(round_idx) / max(1, total_rounds)
    acc_rank = max(cfg.svd_k_min, int(cfg.svd_tau * float(acc))) if acc > 0 else cfg.svd_k_max
    return int(min(int(progress_rank), acc_rank, cfg.svd_k_max))


def selective_svd_reconstruct_topk_dataset(dataset, cfg: TrustAlignConfig, round_idx: int, total_rounds: int, acc: float = 0.0):
    """
    Apply truncated SVD to each sample's [T, top-k] logit matrix.
    """
    rank = dynamic_svd_rank(round_idx, total_rounds, cfg, acc)

    def _map(example):
        logits = torch.tensor(example[PER_STEP_LOGITS], dtype=torch.float32)
        if logits.ndim != 2 or min(logits.shape) <= 1:
            return example
        r = min(rank, min(logits.shape) - 1)
        if r <= 0:
            return example
        try:
            u, s, vh = torch.linalg.svd(logits, full_matrices=False)
            recon = (u[:, :r] * s[:r]) @ vh[:r, :]
            example[PER_STEP_LOGITS] = recon.tolist()
        except RuntimeError:
            pass
        return example

    return dataset.map(_map, load_from_cache_file=False)


@torch.no_grad()
def aggregate_aligned_teachers_with_hspaa(aligned_dataset, cfg: TrustAlignConfig, blending_num: int, round_idx: int = 0):
    """Replace multiple SLM aligned teachers with one trusted semantic teacher.

    Input is the token-aligned FedMKT dataset on the LLM side.  Output keeps the
    FedMKT schema but only exposes ALIGNED_OTHER_*_0, so downstream trainer can
    be invoked with blending_num=1.
    """
    device = cfg.device
    model = HSPAA(cfg).eval()

    def _map(example):
        client_dense = []
        metrics = []
        for i in range(blending_num):
            logits_key = f"{ALIGNED_OTHER_LOGITS}_{i}"
            ids_key = f"{ALIGNED_OTHER_INDICES}_{i}"
            if logits_key not in example or ids_key not in example:
                continue
            dense = sparse_topk_to_dense(
                example[logits_key], example[ids_key], cfg.vocab_size, cfg.temperature, device
            )
            if dense.numel() == 0:
                continue
            client_dense.append(dense)
            metrics.append(_safe_metric(example.get(f"{ALIGNED_OTHER_METRIC}_{i}", []), 0, 0.0))

        if not client_dense:
            return example

        # z_k = E_s(E_p^k(h_k)); weighted aggregation by inverse CE-like metric.
        zs = []
        weights = []
        for idx, dense in enumerate(client_dense):
            h_tilde = model.encode_private(dense.unsqueeze(0))
            z = model.encode_shared(h_tilde)
            zs.append(z)
            weights.append(1.0 / torch.exp(torch.tensor(metrics[idx], device=device)).clamp_min(1e-6))
        weight = torch.stack(weights).float()
        weight = weight / weight.sum().clamp_min(1e-6)
        z_stack = torch.stack(zs, dim=0)  # [K, 1, T, latent]
        z_avg = (z_stack * weight.view(-1, 1, 1, 1)).sum(dim=0)
        model.shared_codec.update_ema(z_avg)
        h_shared = model.decode_shared(z_avg)
        dense_teacher = model.decode_private(h_shared).squeeze(0)
        out_logits, out_ids = dense_to_sparse_topk(dense_teacher, cfg.top_k)

        example[f"{ALIGNED_OTHER_LOGITS}_0"] = out_logits
        example[f"{ALIGNED_OTHER_INDICES}_0"] = out_ids
        example[f"{ALIGNED_OTHER_METRIC}_0"] = [float(sum(metrics) / max(1, len(metrics)))]

        # Remove extra teacher columns to make DataCollatorForFedMKT(blending_num=1) deterministic.
        for i in range(1, blending_num):
            example.pop(f"{ALIGNED_OTHER_LOGITS}_{i}", None)
            example.pop(f"{ALIGNED_OTHER_INDICES}_{i}", None)
            example.pop(f"{ALIGNED_OTHER_METRIC}_{i}", None)
        return example

    return aligned_dataset.map(_map, load_from_cache_file=False)
