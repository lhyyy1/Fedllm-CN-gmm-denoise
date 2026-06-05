# -*- coding: utf-8 -*-
"""Loss-GMM based noisy-label denoising trainer for FedMKT private SLM training.

The implementation is intentionally independent from ``fate_llm.algo.fedmkt`` so the
baseline package can stay untouched.  It follows the common small-loss assumption:
clean samples tend to have lower supervised loss than corrupted-label samples.  A
2-component GMM is fitted on recent per-sample losses and its low-mean component is
used as the clean posterior/weight.
"""

import math
from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from fate_llm.algo.fedmkt.fedmkt import _tokenizer_init_kwargs, _prepare_dispatched_model_for_trainer
from fate_llm.algo.fedmkt.utils.local_metric_logger import log_local_metrics


class GMMNoiseTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with CE/SCE supervised loss and GMM sample reweighting.

    Expected optional batch keys:
      - ``is_noisy``: 0/1 ground-truth synthetic noise flag, used only for metrics.
      - ``clean_idx``/``noisy_idx``: optional metadata, removed before model forward.
    """

    def __init__(self, *args, **kwargs):
        self.gmm_enabled = bool(kwargs.pop("gmm_enabled", True))
        self.gmm_warmup_steps = int(kwargs.pop("gmm_warmup_steps", 20))
        self.gmm_update_interval = int(kwargs.pop("gmm_update_interval", 5))
        self.gmm_history_size = int(kwargs.pop("gmm_history_size", 4096))
        self.gmm_min_samples = int(kwargs.pop("gmm_min_samples", 64))
        self.gmm_clean_threshold = float(kwargs.pop("gmm_clean_threshold", 0.5))
        self.gmm_noisy_weight = float(kwargs.pop("gmm_noisy_weight", 0.2))
        self.gmm_weight_power = float(kwargs.pop("gmm_weight_power", 1.0))
        self.denoise_loss_type = str(kwargs.pop("denoise_loss_type", "ce")).lower()
        self.sce_alpha = float(kwargs.pop("sce_alpha", 1.0))
        self.sce_beta = float(kwargs.pop("sce_beta", 0.1))
        self.rce_epsilon = float(kwargs.pop("rce_epsilon", 1e-4))
        self.loss_log_prefix = kwargs.pop("loss_log_prefix", "client_gmm")
        self.loss_log_every_n_steps = max(1, int(kwargs.pop("loss_log_every_n_steps", 20)))
        self.round_idx = kwargs.pop("round_idx", None)
        tokenizer = kwargs.pop("tokenizer", None)
        kwargs.update(_tokenizer_init_kwargs(Seq2SeqTrainer, tokenizer))
        _prepare_dispatched_model_for_trainer(kwargs.get("model", args[0] if args else None))
        super().__init__(*args, **kwargs)
        self._loss_history = deque(maxlen=max(128, self.gmm_history_size))
        self._gmm_params: Optional[Dict[str, float]] = None
        self._loss_log_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        is_noisy = inputs.pop("is_noisy", None)
        # Pure analysis/debug columns.  They must not be passed to HF models.
        inputs.pop("clean_idx", None)
        inputs.pop("noisy_idx", None)
        inputs.pop("train_idx", None)
        inputs.pop("noise_source", None)
        inputs.pop("noise_seed", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if labels is None:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                loss = outputs[0]
            return (loss, outputs) if return_outputs else loss

        sample_ce, sample_rce = self._sample_lm_losses(model, outputs, labels)
        if self.denoise_loss_type in {"sce", "symmetric_ce", "symmetric-ce"}:
            sample_loss = self.sce_alpha * sample_ce + self.sce_beta * sample_rce
        elif self.denoise_loss_type == "ce":
            sample_loss = sample_ce
        else:
            raise ValueError(f"unsupported denoise_loss_type={self.denoise_loss_type!r}; use ce or symmetric_ce")

        clean_prob = self._estimate_clean_probability(sample_ce.detach())
        weights = self.gmm_noisy_weight + (1.0 - self.gmm_noisy_weight) * clean_prob
        if self.gmm_weight_power != 1.0:
            weights = weights.clamp_min(1e-6).pow(self.gmm_weight_power)
        if not self.gmm_enabled:
            weights = torch.ones_like(sample_loss)

        loss = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        self._log_gmm_metrics(loss, sample_ce, sample_rce, weights, clean_prob, is_noisy)
        return (loss, outputs) if return_outputs else loss

    def _sample_lm_losses(self, model, outputs, labels):
        logits = outputs.logits
        if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()

        vocab_size = shift_logits.size(-1)
        flat_labels = shift_labels.view(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        token_ce = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100).view(shift_labels.size())
        mask = shift_labels.ne(-100).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        sample_ce = (token_ce * mask).sum(dim=1) / denom

        # Reverse CE used by symmetric CE.  With clipped one-hot labels y, RCE is
        # -sum_c p_c log(y_c).  Since log(y_true)=0 and log(y_other)=log(eps),
        # RCE = -(1 - p_true) * log(eps) on valid tokens.
        with torch.no_grad():
            safe_labels = flat_labels.clone()
            safe_labels[safe_labels < 0] = 0
        probs = torch.softmax(flat_logits, dim=-1)
        true_prob = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1).view(shift_labels.size())
        token_rce = -(1.0 - true_prob) * math.log(max(self.rce_epsilon, 1e-12))
        sample_rce = (token_rce * mask).sum(dim=1) / denom
        return sample_ce, sample_rce

    def _estimate_clean_probability(self, sample_ce: torch.Tensor) -> torch.Tensor:
        device = sample_ce.device
        if self.gmm_enabled:
            self._loss_history.extend(float(x) for x in sample_ce.float().cpu().tolist() if math.isfinite(float(x)))
            step = int(getattr(self.state, "global_step", 0) or 0)
            if (
                step >= self.gmm_warmup_steps
                and len(self._loss_history) >= self.gmm_min_samples
                and (self._gmm_params is None or step % self.gmm_update_interval == 0)
            ):
                self._gmm_params = self._fit_loss_gmm(list(self._loss_history))

        if not self.gmm_enabled or self._gmm_params is None:
            return torch.ones_like(sample_ce, device=device)

        x = sample_ce.float()
        p = self._posterior_low_mean_component(x, self._gmm_params)
        return p.to(device=device, dtype=sample_ce.dtype).clamp(0.0, 1.0)

    @staticmethod
    def _fit_loss_gmm(values):
        import numpy as np
        arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        arr = arr[np.isfinite(arr[:, 0])]
        if arr.shape[0] < 8:
            return None
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-6, random_state=0)
            gmm.fit(arr.reshape(-1, 1))
            means = gmm.means_.reshape(-1)
            variances = gmm.covariances_.reshape(2, -1)[:, 0]
            weights = gmm.weights_.reshape(-1)
        except Exception:
            # Lightweight EM fallback when sklearn is unavailable.
            q25, q75 = np.percentile(arr, [25, 75])
            means = np.array([q25, q75], dtype=np.float64)
            variances = np.array([np.var(arr) + 1e-6, np.var(arr) + 1e-6], dtype=np.float64)
            weights = np.array([0.5, 0.5], dtype=np.float64)
            flat = arr.reshape(-1)
            for _ in range(20):
                probs = []
                for k in range(2):
                    var = max(float(variances[k]), 1e-6)
                    probs.append(weights[k] / math.sqrt(2.0 * math.pi * var) * np.exp(-0.5 * (flat - means[k]) ** 2 / var))
                resp = np.stack(probs, axis=1)
                resp = resp / np.clip(resp.sum(axis=1, keepdims=True), 1e-12, None)
                nk = resp.sum(axis=0) + 1e-12
                weights = nk / len(flat)
                means = (resp * flat[:, None]).sum(axis=0) / nk
                variances = (resp * (flat[:, None] - means[None, :]) ** 2).sum(axis=0) / nk + 1e-6
        low = int(np.argmin(means))
        return {
            "low": low,
            "mean0": float(means[0]),
            "mean1": float(means[1]),
            "var0": float(max(variances[0], 1e-6)),
            "var1": float(max(variances[1], 1e-6)),
            "weight0": float(max(weights[0], 1e-6)),
            "weight1": float(max(weights[1], 1e-6)),
        }

    @staticmethod
    def _posterior_low_mean_component(x: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
        comps = []
        for k in (0, 1):
            mean = x.new_tensor(params[f"mean{k}"])
            var = x.new_tensor(params[f"var{k}"]).clamp_min(1e-6)
            prior = x.new_tensor(params[f"weight{k}"]).clamp_min(1e-6)
            logp = torch.log(prior) - 0.5 * (torch.log(2.0 * torch.pi * var) + (x - mean).pow(2) / var)
            comps.append(logp)
        log_probs = torch.stack(comps, dim=1)
        probs = torch.softmax(log_probs, dim=1)
        return probs[:, int(params["low"])]

    def _log_gmm_metrics(self, total_loss, sample_ce, sample_rce, weights, clean_prob, is_noisy):
        self._loss_log_count += 1
        if self._loss_log_count % self.loss_log_every_n_steps != 0:
            return
        round_value = -1 if self.round_idx is None else int(self.round_idx)
        loss_step = round_value * 100000 + int(getattr(self.state, "global_step", 0) or 0)
        metrics = {
            f"{self.loss_log_prefix}/fedmkt_round": round_value,
            f"{self.loss_log_prefix}/fedmkt_loss_step": int(loss_step),
            f"{self.loss_log_prefix}/total_loss": float(total_loss.detach().float().cpu()),
            f"{self.loss_log_prefix}/ce_loss_mean": float(sample_ce.detach().float().mean().cpu()),
            f"{self.loss_log_prefix}/rce_loss_mean": float(sample_rce.detach().float().mean().cpu()),
            f"{self.loss_log_prefix}/sample_weight_mean": float(weights.detach().float().mean().cpu()),
            f"{self.loss_log_prefix}/clean_prob_mean": float(clean_prob.detach().float().mean().cpu()),
        }
        if self._gmm_params is not None:
            metrics.update({
                f"{self.loss_log_prefix}/gmm_low_mean": min(self._gmm_params["mean0"], self._gmm_params["mean1"]),
                f"{self.loss_log_prefix}/gmm_high_mean": max(self._gmm_params["mean0"], self._gmm_params["mean1"]),
            })
        if is_noisy is not None:
            y = is_noisy.detach().float().view(-1).to(clean_prob.device)
            pred_noisy = (clean_prob < self.gmm_clean_threshold).float()
            if y.numel() == pred_noisy.numel() and y.numel() > 0:
                tp = ((pred_noisy == 1) & (y == 1)).float().sum()
                fp = ((pred_noisy == 1) & (y == 0)).float().sum()
                fn = ((pred_noisy == 0) & (y == 1)).float().sum()
                correct = (pred_noisy == y).float().mean()
                metrics.update({
                    f"{self.loss_log_prefix}/noise_detect_acc": float(correct.detach().cpu()),
                    f"{self.loss_log_prefix}/noise_detect_precision": float((tp / (tp + fp).clamp_min(1.0)).detach().cpu()),
                    f"{self.loss_log_prefix}/noise_detect_recall": float((tp / (tp + fn).clamp_min(1.0)).detach().cpu()),
                    f"{self.loss_log_prefix}/batch_noise_rate": float(y.mean().detach().cpu()),
                })
        self._emit_metrics(metrics)

    @staticmethod
    def _emit_metrics(metrics):
        log_local_metrics(metrics)
        try:
            import wandb
            if wandb.run is not None and getattr(wandb.run.settings, "mode", None) != "disabled":
                wandb.log(metrics)
        except Exception:
            pass
        try:
            import swanlab
            if getattr(swanlab, "log", None) is not None:
                swanlab.log(metrics)
        except Exception:
            pass
        print("[GMM denoise] " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)
