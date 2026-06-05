from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import List, Optional, Union

import datasets
import torch
import transformers
from transformers.modeling_utils import unwrap_model

from fate_llm.algo.fedmkt.fedmkt import FedMKTLLM, FedMKTSLM, FedMKTTrainingArguments
from fate_llm.algo.fedmkt.fedmkt_data_collator import DataCollatorForFedMKT
from fate_llm.algo.fedmkt.fedmkt_trainer import FedMKTTrainer
from fate_llm.algo.fedmkt.token_alignment.token_align import token_align
from fate_llm.algo.fedmkt.utils.dataset_sync_util import sync_dataset
from fate_llm.algo.fedmkt.utils.generate_logit_utils import generate_pub_data_logits
from fate_llm.trainer.seq2seq_trainer import Seq2SeqTrainingArguments

from .trust_autoencoder import (
    TrustAlignConfig,
    aggregate_aligned_teachers_with_hspaa,
    selective_svd_reconstruct_topk_dataset,
)


@dataclass
class TrustedFedMKTTrainingArguments(FedMKTTrainingArguments):
    # Master switch.
    use_trust_align: bool = field(default=True)

    # HSPAA module sizes.
    trust_latent_dim: int = field(default=256)
    trust_hidden_dim: int = field(default=512)
    trust_num_heads: int = field(default=4)
    trust_ema_decay: float = field(default=0.95)
    trust_private_reg_lambda: float = field(default=1e-4)

    # Cloud selective SVD reconstruction.
    trust_use_svd: bool = field(default=True)
    trust_svd_k_min: int = field(default=8)
    trust_svd_k_max: int = field(default=64)
    trust_svd_tau: float = field(default=128.0)

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_dict_without_extra_args(self):
        d = super().to_dict_without_extra_args()
        for k in list(d.keys()):
            if k.startswith("trust_") or k == "use_trust_align":
                d.pop(k, None)
        return d

    def trust_cfg(self) -> TrustAlignConfig:
        if self.vocab_size is None:
            raise ValueError("TrustedFedMKT requires training_args.vocab_size")
        return TrustAlignConfig(
            vocab_size=int(self.vocab_size),
            top_k=int(self.top_k_logits_keep),
            latent_dim=int(self.trust_latent_dim),
            hidden_dim=int(self.trust_hidden_dim),
            num_heads=int(self.trust_num_heads),
            ema_decay=float(self.trust_ema_decay),
            svd_k_min=int(self.trust_svd_k_min),
            svd_k_max=int(self.trust_svd_k_max),
            svd_tau=float(self.trust_svd_tau),
            private_reg_lambda=float(self.trust_private_reg_lambda),
            temperature=float(self.distill_temperature),
            device=str(self.device),
        )


class TrustedFedMKTLLM(FedMKTLLM):
    """Server/arbiter side: aggregate SLM teachers in a trusted latent space."""

    training_args: TrustedFedMKTTrainingArguments

    def on_epoch_begin(self, iter_ctx, epoch_idx, previous_pub_dataset):
        aligned_dataset = super().on_epoch_begin(iter_ctx, epoch_idx, previous_pub_dataset)
        if not self.training_args.use_trust_align:
            return aligned_dataset
        blending_num = len(self.slm_tokenizers)
        if blending_num <= 0:
            return aligned_dataset
        return aggregate_aligned_teachers_with_hspaa(
            aligned_dataset=aligned_dataset,
            cfg=self.training_args.trust_cfg(),
            blending_num=blending_num,
            round_idx=epoch_idx,
        )

    def on_epoch_end(self, iter_ctx, epoch_idx):
        llm_pub_logits = super().on_epoch_end(iter_ctx, epoch_idx)
        if (
            self.training_args.use_trust_align
            and self.training_args.trust_use_svd
            and llm_pub_logits is not None
        ):
            llm_pub_logits = selective_svd_reconstruct_topk_dataset(
                llm_pub_logits,
                cfg=self.training_args.trust_cfg(),
                round_idx=epoch_idx,
                total_rounds=self.training_args.global_epochs,
                acc=0.0,
            )
            # Re-send reconstructed teacher knowledge to clients.
            if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
                iter_ctx.guest.put("llm_pub_logits", llm_pub_logits.to_dict())
                if len(self.slm_tokenizers) > 1:
                    iter_ctx.hosts.put("llm_pub_logits", llm_pub_logits.to_dict())
        return llm_pub_logits

    def train(self):
        global_epochs = self.training_args.global_epochs
        previous_pub_logits = None
        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            aligned_train_set = self.on_epoch_begin(iter_ctx, i, previous_pub_logits)
            if self.training_args.llm_training:
                public_data_training_args = self._get_pub_data_kd_training_args()
                blending_num = 1 if self.training_args.use_trust_align else len(self.slm_tokenizers)
                trainer = FedMKTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=public_data_training_args,
                    train_dataset=aligned_train_set,
                    eval_dataset=self.val_set,
                    data_collator=DataCollatorForFedMKT(
                        self.tokenizer,
                        padding="max_length",
                        max_length=max(len(d["input_ids"]) for d in aligned_train_set),
                        blending_num=blending_num,
                        vocab_size=self.training_args.vocab_size,
                        dtype=next(self.model.parameters()).dtype,
                        distill_temperature=self.training_args.distill_temperature,
                    ),
                    blending_num=blending_num,
                    lm_loss_weight=self.training_args.kd_alpha,
                    distill_loss_type=self.training_args.distill_loss_type,
                    distill_strategy=self.training_args.distill_strategy,
                    loss_log_prefix="trusted_server",
                )
                trainer.train()
                self.model = unwrap_model(trainer.model)
            previous_pub_logits = self.on_epoch_end(iter_ctx, i)


class TrustedFedMKTSLM(FedMKTSLM):
    """Client side: same FedMKT flow, consuming reconstructed trusted LLM logits."""

    training_args: TrustedFedMKTTrainingArguments

    def train(self):
        # FedMKTSLM.train already performs private data training, public-logit
        # generation, token alignment, and KD from LLM.  The server sends SVD-
        # reconstructed/trusted LLM logits under the same llm_pub_logits key, so
        # no communication protocol change is needed on the client.
        return super().train()
