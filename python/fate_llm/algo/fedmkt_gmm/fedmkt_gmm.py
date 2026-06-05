# -*- coding: utf-8 -*-
"""FedMKT variant with GMM-based noisy-label denoising.

This module reuses the baseline FedMKT components but does not modify
``python/fate_llm/algo/fedmkt``.  Denoising is applied to SLM private-data
training, where synthetic label noise is injected into client data.
"""

import logging
from dataclasses import dataclass, field

import datasets
import transformers
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model

from fate_llm.algo.fedmkt.fedmkt import (
    FedMKTTrainingArguments,
    FedMKTSLM,
    FedMKTLLM,
    _prepare_dispatched_model_for_trainer,
    _log_model_runtime_device,
    _log_round_arc_accuracy,
    sync_dataset,
    token_align,
    generate_pub_data_logits,
    DataCollatorForFedMKT,
)
from fate_llm.algo.fedmkt_gmm.gmm_trainer import GMMNoiseTrainer

logger = logging.getLogger(__name__)


@dataclass
class FedMKTGMMTrainingArguments(FedMKTTrainingArguments):
    gmm_enabled: bool = field(default=True)
    gmm_warmup_steps: int = field(default=20)
    gmm_update_interval: int = field(default=5)
    gmm_history_size: int = field(default=4096)
    gmm_min_samples: int = field(default=64)
    gmm_clean_threshold: float = field(default=0.5)
    gmm_noisy_weight: float = field(default=0.2)
    gmm_weight_power: float = field(default=1.0)
    denoise_loss_type: str = field(default="ce")
    sce_alpha: float = field(default=1.0)
    sce_beta: float = field(default=0.1)
    rce_epsilon: float = field(default=1e-4)
    denoise_loss_log_every_n_steps: int = field(default=20)

    def to_dict_without_extra_args(self):
        args_dict = super().to_dict_without_extra_args()
        for key in (
            "gmm_enabled",
            "gmm_warmup_steps",
            "gmm_update_interval",
            "gmm_history_size",
            "gmm_min_samples",
            "gmm_clean_threshold",
            "gmm_noisy_weight",
            "gmm_weight_power",
            "denoise_loss_type",
            "sce_alpha",
            "sce_beta",
            "rce_epsilon",
            "denoise_loss_log_every_n_steps",
        ):
            args_dict.pop(key, None)
        return args_dict


class FedMKTGMMSLM(FedMKTSLM):
    """FedMKT SLM client using GMMNoiseTrainer for private noisy data."""

    def train(self):
        global_epochs = self.training_args.global_epochs
        llm_pub_logits = None
        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            logger.info(f"begin {i}-th global kd process with GMM denoising")
            self._fedmkt_current_round = i
            priv_data_training_args = self._get_priv_data_training_args()
            _prepare_dispatched_model_for_trainer(self.model)

            priv_trainer = GMMNoiseTrainer(
                model=self.model,
                data_collator=self.priv_data_collator,
                train_dataset=self.priv_train_set,
                args=priv_data_training_args,
                model_init=self.model_init if not i else None,
                compute_metrics=self.compute_metrics,
                callbacks=self.callbacks,
                optimizers=(self.priv_optimizer, self.priv_scheduler),
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
                tokenizer=self.tokenizer,
                gmm_enabled=self.training_args.gmm_enabled,
                gmm_warmup_steps=self.training_args.gmm_warmup_steps,
                gmm_update_interval=self.training_args.gmm_update_interval,
                gmm_history_size=self.training_args.gmm_history_size,
                gmm_min_samples=self.training_args.gmm_min_samples,
                gmm_clean_threshold=self.training_args.gmm_clean_threshold,
                gmm_noisy_weight=self.training_args.gmm_noisy_weight,
                gmm_weight_power=self.training_args.gmm_weight_power,
                denoise_loss_type=self.training_args.denoise_loss_type,
                sce_alpha=self.training_args.sce_alpha,
                sce_beta=self.training_args.sce_beta,
                rce_epsilon=self.training_args.rce_epsilon,
                loss_log_prefix=f"slm{getattr(self, 'slm_idx', 'x')}_private_gmm",
                loss_log_every_n_steps=self.training_args.denoise_loss_log_every_n_steps,
                round_idx=i,
            )
            _log_model_runtime_device(f"client_gmm_priv_trainer_round_{i}", priv_trainer.model, priv_data_training_args)

            logger.info(f"begin {i}-th private noisy data training process with GMM denoising")
            priv_trainer.train()
            self.model = unwrap_model(priv_trainer.model)

            logger.info(f"begin {i}-th public logits generation process")
            if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
                slm_pub_logits = self.pub_train_set.map(
                    generate_pub_data_logits,
                    batched=True,
                    batch_size=self.training_args.per_device_train_batch_size,
                    num_proc=None,
                    load_from_cache_file=False,
                    fn_kwargs={
                        "model": self.model,
                        "training_args": self.training_args,
                        "data_collator": transformers.DataCollatorForSeq2Seq(self.tokenizer),
                    },
                )

                if self.training_args.world_size > 1:
                    logger.info("sync slm_pub_logits")
                    sync_dataset(slm_pub_logits, self.training_args.local_rank, self.training_args.world_size, self.training_args.device)

                if self.training_args.llm_training:
                    iter_ctx.arbiter.put("slm_pub_logits", slm_pub_logits.to_dict())

                if self.training_args.llm_training or not i:
                    llm_pub_logits = datasets.Dataset.from_dict(iter_ctx.arbiter.get("llm_pub_logits"))
                    if self.training_args.world_size > 1:
                        logger.info("sync llm_pub_logits")
                        sync_dataset(llm_pub_logits, self.training_args.local_rank, self.training_args.world_size, self.training_args.device)
            else:
                slm_pub_logits = sync_dataset(None, self.training_args.local_rank, self.training_args.world_size, self.training_args.device)
                if self.training_args.llm_training or not i:
                    llm_pub_logits = sync_dataset(None, self.training_args.local_rank, self.training_args.world_size, self.training_args.device)

            logger.info(f"begin {i}-th token alignment process")
            aligned_dataset = token_align(
                base_model_logits_datasets=slm_pub_logits,
                blending_model_logits_dataset=llm_pub_logits,
                base_tokenizer=self.tokenizer,
                blending_tokenizer=self.llm_tokenizer,
                blending_to_base_mapping=self.llm_to_slm_vocab_mapping,
                blending_model_index=0,
                skip_align=self.training_args.skip_align,
                align_strategy=self.training_args.token_align_strategy,
            )

            logger.info(f"begin {i}-th public logits kd process")
            fedmkt_trainer = self._init_trainer_for_distill(aligned_dataset)
            _log_model_runtime_device(f"client_kd_trainer_round_{i}", fedmkt_trainer.model, fedmkt_trainer.args)
            fedmkt_trainer.train()
            self.model = unwrap_model(fedmkt_trainer.model)

            if self.training_args.post_fedavg and (i + 1) % self.fed_args.aggregate_freq == 0:
                self.aggregator.model_aggregation(iter_ctx, self.model)

            _log_round_arc_accuracy(self.model, self.tokenizer, i, "client_gmm")


__all__ = ["FedMKTGMMTrainingArguments", "FedMKTGMMSLM", "FedMKTLLM"]
