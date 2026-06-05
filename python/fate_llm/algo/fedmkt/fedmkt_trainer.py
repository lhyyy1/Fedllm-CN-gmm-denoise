#
# NOTE: The implementations of FedMKTTrainer is modified from FuseAI/FuseLLM
# Copyright FuseAI
#
#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import inspect
import torch
from torch.nn.functional import kl_div, log_softmax, cross_entropy
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from fate_llm.algo.fedmkt.utils.vars_define import (
    SELF_TARGET_DIST,
    OTHER_TARGET_DIST,
    ALIGNED_OTHER_METRIC,
    METRIC,
)
from fate_llm.algo.fedmkt.utils.local_metric_logger import log_local_metrics

logger = logging.getLogger(__name__)


def _tokenizer_init_kwargs(trainer_cls, tokenizer):
    if tokenizer is None:
        return {}

    init_params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in init_params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _prepare_dispatched_model_for_trainer(model):
    if model is None:
        return

    has_device_map = getattr(model, "hf_device_map", None) is not None
    has_meta_params = any(param.is_meta for param in model.parameters())
    if has_device_map or has_meta_params:
        model.is_parallelizable = True
        model.model_parallel = True


class FedMKTTrainer(Seq2SeqTrainer):
    """
    modified from https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/trainer.py#L22
    """
    blending_num: int = 2
    distill_loss_type: str = "ce"
    lm_loss_weight: float = 0.9
    distill_strategy = "greater"

    def __init__(self, *args, **kwargs):
        blending_num = kwargs.pop("blending_num", 1)
        distill_loss_type = kwargs.pop("distill_loss_type", "ce")
        lm_loss_weight = kwargs.pop("lm_loss_weight", 0.9)
        distill_strategy = kwargs.pop("distill_strategy", "greater")
        loss_log_prefix = kwargs.pop("loss_log_prefix", "fedmkt")
        loss_log_every_n_steps = kwargs.pop("loss_log_every_n_steps", 20)
        round_idx = kwargs.pop("round_idx", None)
        tokenizer = kwargs.pop("tokenizer", None)
        kwargs.update(_tokenizer_init_kwargs(Seq2SeqTrainer, tokenizer))
        _prepare_dispatched_model_for_trainer(kwargs.get("model", args[0] if args else None))
        super(FedMKTTrainer, self).__init__(*args, **kwargs)
        self.blending_num = blending_num
        self.distill_loss_type = distill_loss_type
        self.lm_loss_weight = lm_loss_weight
        self.distill_strategy = distill_strategy
        self.loss_log_prefix = loss_log_prefix
        self.loss_log_every_n_steps = max(1, int(loss_log_every_n_steps))
        self.round_idx = round_idx
        self._loss_log_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        base_target_dist = inputs.pop(SELF_TARGET_DIST)
        base_metric = inputs.pop(METRIC)

        aligned_target_dists = []
        aligned_metrics = []
        for i in range(self.blending_num):
            aligned_target_dists.append(inputs.pop(f"{OTHER_TARGET_DIST}_{i}"))
            aligned_metrics.append(inputs.pop(f"{ALIGNED_OTHER_METRIC}_{i}"))

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        past_index = getattr(self.args, "past_index", -1)
        if past_index >= 0:
            self._past = outputs[past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        batch_size, seq_len, vocab_size = outputs["logits"].size(0), outputs["logits"].size(1), outputs["logits"].size(2)

        #指标越小（模型预测越准），reward 就越大。相当于一种 动态质量感知机制：更可靠的模型知识会被赋予更高权重
        aligned_rewards = []
        for i in range(self.blending_num):
            aligned_rewards.append((1 / torch.exp(torch.tensor(aligned_metrics[i], dtype=torch.bfloat16))).to(loss.device))

        base_reward = (1 / torch.exp(torch.tensor(base_metric, dtype=torch.bfloat16))).to(loss.device)

        #把所有 SLM 的分布堆在一起。在每个位置上，选择 reward 最大的那个模型的分布 作为 teacher。
        #aligned_target_dist: [batch, seq_len, vocab_size_llm]
        if self.distill_strategy == "greater":
            rewards = torch.stack([base_reward] + aligned_rewards, dim=1)
            best_teacher_indices = torch.argmax(rewards, dim=1)
            target_dist_candidates = [base_target_dist] + aligned_target_dists
            target_dist = torch.empty_like(base_target_dist)
            for teacher_idx, candidate_dist in enumerate(target_dist_candidates):
                row_mask = best_teacher_indices == teacher_idx
                if row_mask.any():
                    target_dist[row_mask] = candidate_dist[row_mask]
        #对所有模型的分布做 softmax 加权平均
        elif self.distill_strategy == "weighted_mean":
            weights = torch.stack(
                [base_reward] + aligned_rewards, dim=1
            )
            normalized_weights = torch.softmax(weights, dim=1)
            weight_labels = normalized_weights[:, 0].unsqueeze(1).unsqueeze(2) * base_target_dist
            for i in range(self.blending_num):
                weight_labels += normalized_weights[:, i + 1].unsqueeze(1).unsqueeze(2) * aligned_target_dists[i]

            target_dist = (
                weight_labels
            )
        else:
            raise ValueError(f"distill_strategy={self.distill_strategy}")

        if self.distill_loss_type == "ce":
            loss_lm = cross_entropy(
                input=outputs["logits"].view(-1, vocab_size),
                target=target_dist.view(-1, vocab_size),
                reduction="none",
            ).view(batch_size, -1)
        elif self.distill_loss_type == "kl":
            loss_lm = kl_div(
                input=log_softmax(outputs["logits"], dim=-1),
                target=target_dist,
                log_target=False,
                reduction="none").sum(dim=-1)
        else:
            raise ValueError(f"Not implement distill_loss_type={self.distill_loss_type}")

        loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs["attention_mask"].sum()
        supervised_loss = loss
        distill_loss = loss_lm
        loss = self.lm_loss_weight * supervised_loss + (1.0 - self.lm_loss_weight) * distill_loss
        self._log_loss_parts(loss, supervised_loss, distill_loss)

        return (loss, outputs) if return_outputs else loss

    def _log_loss_parts(self, total_loss, supervised_loss, distill_loss):
        self._loss_log_count += 1
        if self._loss_log_count % self.loss_log_every_n_steps != 0:
            return

        total_value = float(total_loss.detach().float().cpu())
        supervised_value = float(supervised_loss.detach().float().cpu())
        distill_value = float(distill_loss.detach().float().cpu())
        supervised_weighted = self.lm_loss_weight * supervised_value
        distill_weighted = (1.0 - self.lm_loss_weight) * distill_value
        round_value = -1 if self.round_idx is None else int(self.round_idx)
        loss_step = self.state.global_step if self.round_idx is None else round_value * 100000 + self.state.global_step

        metrics = {
            f"{self.loss_log_prefix}/fedmkt_round": round_value,
            f"{self.loss_log_prefix}/fedmkt_loss_step": int(loss_step),
            f"{self.loss_log_prefix}/trainer_global_step": int(self.state.global_step),
            f"{self.loss_log_prefix}/loss_log_count": int(self._loss_log_count),
            f"{self.loss_log_prefix}/total_loss": total_value,
            f"{self.loss_log_prefix}/supervised_lm_loss": supervised_value,
            f"{self.loss_log_prefix}/distill_loss": distill_value,
            f"{self.loss_log_prefix}/weighted_supervised_lm_loss": supervised_weighted,
            f"{self.loss_log_prefix}/weighted_distill_loss": distill_weighted,
            f"{self.loss_log_prefix}/lm_loss_weight": float(self.lm_loss_weight),
        }
        try:
            import wandb
            if wandb.run is not None and wandb.run.settings.mode != "disabled":
                wandb.log(metrics)
        except Exception:
            pass
        try:
            import swanlab
            if getattr(swanlab, "log", None) is not None:
                swanlab.log(metrics)
        except Exception:
            pass
        log_local_metrics(metrics)

        print(
            f"[FedMKT loss][{self.loss_log_prefix}] "
            f"step={self.state.global_step} "
            f"total={total_value:.6f} "
            f"supervised={supervised_value:.6f} "
            f"distill={distill_value:.6f} "
            f"weighted_supervised={supervised_weighted:.6f} "
            f"weighted_distill={distill_weighted:.6f} "
            f"lm_loss_weight={self.lm_loss_weight:.4f}",
            flush=True,
        )
