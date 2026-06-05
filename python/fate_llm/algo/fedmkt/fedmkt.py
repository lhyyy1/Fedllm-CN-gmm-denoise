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
import torch
import logging
import datasets
import os
import inspect
from dataclasses import dataclass, field

import transformers

from ...trainer.seq2seq_trainer import Seq2SeqTrainingArguments
from typing import Dict, Optional, List, Callable, Union
from fate.arch import Context
from fate.ml.nn.trainer.trainer_base import FedArguments
from torch.utils.data import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers import PreTrainedTokenizer
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_utils import unwrap_model
from fate_llm.algo.fedmkt.token_alignment.token_align import token_align
from fate_llm.algo.fedmkt.utils.generate_logit_utils import generate_pub_data_logits
from fate.ml.aggregator import AggregatorClientWrapper, AggregatorServerWrapper
from fate_llm.algo.fedmkt.fedmkt_trainer import FedMKTTrainer
from fate_llm.algo.fedmkt.fedmkt_data_collator import DataCollatorForFedMKT
from fate_llm.algo.fedmkt.utils.dataset_sync_util import sync_dataset
from fate_llm.algo.fedmkt.utils.local_metric_logger import local_tracking_enabled, log_local_metrics
from fate_llm.algo.fedmkt.mmlcc import aggregate_aligned_slm_teachers_dataset


logger = logging.getLogger(__name__)


def _tokenizer_init_kwargs(trainer_cls, tokenizer):
    if tokenizer is None:
        return {}

    init_params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in init_params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _get_hf_device_map(model):
    seen = set()
    candidates = [model]
    while candidates:
        candidate = candidates.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        device_map = getattr(candidate, "hf_device_map", None)
        if device_map is not None:
            return device_map

        for attr_name in ("module", "_pe_lm", "base_model", "model"):
            child = getattr(candidate, attr_name, None)
            if child is not None and child is not candidate:
                candidates.append(child)

    return None


def _has_meta_parameters(model):
    return any(param.is_meta for param in model.parameters())


def _move_model_to_training_device(model, device):
    if device.type != "cuda":
        return

    device_map = _get_hf_device_map(model)
    if device_map is not None:
        logger.info(f"skip moving model to {device}; model is already dispatched by hf device_map={device_map}")
        return

    if _has_meta_parameters(model):
        logger.info(f"skip moving model to {device}; model contains meta tensors managed by lazy loading/offload")
        return

    model.cuda(device)


def _prepare_dispatched_model_for_trainer(model):
    if model is None:
        return

    has_device_map = _get_hf_device_map(model) is not None
    has_meta_params = _has_meta_parameters(model)
    if has_device_map or has_meta_params:
        model.is_parallelizable = True
        model.model_parallel = True


def _log_model_runtime_device(prefix, model, training_args=None):
    try:
        first_param_device = next(model.parameters()).device
    except StopIteration:
        first_param_device = "no_parameters"
    except Exception as exc:
        first_param_device = f"unavailable:{exc}"
    args_device = getattr(training_args, "device", None)
    local_rank = getattr(training_args, "local_rank", None)
    logger.info(
        f"[model-device][{prefix}] "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"training_args.device={args_device} "
        f"local_rank={local_rank} "
        f"first_parameter_device={first_param_device} "
        f"hf_device_map={_get_hf_device_map(model)}"
    )


def _force_training_args_device_from_env(training_args):
    forced_device = os.environ.get("FEDMKT_FORCE_CUDA_DEVICE")
    if forced_device in {None, ""}:
        return training_args
    device = torch.device(f"cuda:{int(forced_device)}")
    torch.cuda.set_device(device)
    training_args.__dict__["_setup_devices"] = device
    training_args._n_gpu = 1
    logger.info(
        f"[training-args-device] forced training_args.device={device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    return training_args


def _wandb_is_active(wandb_module):
    return (
        wandb_module is not None
        and wandb_module.run is not None
        and getattr(wandb_module.run.settings, "mode", None) != "disabled"
    )


def _swanlab_log(metrics):
    log_local_metrics(metrics)
    try:
        import swanlab
        if getattr(swanlab, "log", None) is not None:
            swanlab.log(metrics)
    except Exception:
        pass


def _get_wandb_eval_max_examples():
    raw_value = os.environ.get("FEDMKT_WANDB_EVAL_MAX_EXAMPLES", "200")
    if raw_value.strip() in {"", "0", "none", "None", "NONE"}:
        return None
    return int(raw_value)

def _log_round_arc_accuracy(model, tokenizer, round_idx: int, prefix: str):
    wandb_active = False
    try:
        import wandb
        wandb_active = _wandb_is_active(wandb)
    except ImportError:
        wandb = None

    swanlab_mode = os.environ.get("SWANLAB_MODE", "disabled").lower()
    swanlab_active = swanlab_mode not in {"", "disabled", "disable", "false", "0", "none"}
    if not wandb_active and not swanlab_active and not local_tracking_enabled():
        return

    from fate_llm.evaluate.arc_eval import evaluate_arc_mc_accuracy

    eval_split = os.environ.get("FEDMKT_WANDB_EVAL_SPLIT", "validation")
    max_examples = _get_wandb_eval_max_examples()
    print(
        f"[tracking] begin {prefix} round={round_idx} ARC eval "
        f"split={eval_split} max_examples={max_examples}",
        flush=True,
    )

    metrics = evaluate_arc_mc_accuracy(
        model=model,
        tokenizer=tokenizer,
        split=eval_split,
        max_examples=max_examples,
    )

    log_metrics = {
        f"{prefix}/fedmkt_round": int(round_idx),
        f"{prefix}/arc_mc_accuracy": metrics["arc_mc_accuracy"],
        f"{prefix}/arc_mc_avg_choice_loss": metrics["arc_mc_avg_choice_loss"],
        f"{prefix}/arc_mc_num_examples": metrics["arc_mc_num_examples"],
    }
    if wandb_active:
        wandb.log(log_metrics)
    _swanlab_log(log_metrics)
    print(
        f"[tracking] logged {prefix} round={round_idx} "
        f"arc_mc_accuracy={metrics['arc_mc_accuracy']:.6f} "
        f"arc_mc_avg_choice_loss={metrics['arc_mc_avg_choice_loss']:.6f} "
        f"arc_mc_num_examples={metrics['arc_mc_num_examples']}",
        flush=True,
    )


@dataclass
class FedMKTTrainingArguments(Seq2SeqTrainingArguments):
    """
    selection metric type
    """
    metric_type: str = field(default="ce")

    """
    top-k logits select params
    """
    top_k_logits_keep: int = field(default=128)
    top_k_strategy: str = field(default="highest")

    """
    distillation params
    """
    distill_loss_type: str = field(default="ce")
    kd_alpha: float = field(default=0.9)
    distill_temperature: float = field(default=1.0)
    server_public_data_local_epoch: int = field(default=1)
    client_public_data_local_epoch: int = field(default=1)
    client_priv_data_local_epoch: int = field(default=1)
    distill_strategy: str = field(default="greater")
    global_epochs: int = field(default=1)

    """
    token-alignment params
    """
    skip_align: bool = field(default=False)
    token_align_strategy: str = field(default="dtw")
    vocab_mapping_paths: Union[str, List[str]] = field(default=None)
    vocab_size: int = field(default=None)

    """
    homo training params
    """
    post_fedavg: bool = field(default=False)

    """
    slm training only
    """
    llm_training: bool = field(default=True)

    """
    MMLCC-style aligned SLM teacher aggregation.
    """
    use_mmlcc_aggregation: bool = field(default=False)
    mmlcc_probability_epsilon: float = field(default=1e-12)
    mmlcc_num_blocks: int = field(default=1)
    mmlcc_privacy_guarantee: int = field(default=1)
    mmlcc_beta_radius: float = field(default=1.15)
    mmlcc_noise_sigma: float = field(default=1.0)
    mmlcc_noise_clip_theta: float = field(default=6.0)
    mmlcc_seed: int = field(default=42)

    def to_dict(self):
        from dataclasses import fields
        from enum import Enum
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_dict_without_extra_args(self):
        args_dict = self.to_dict()
        args_dict.pop("metric_type")
        args_dict.pop("top_k_logits_keep")
        args_dict.pop("top_k_strategy")

        args_dict.pop("distill_loss_type")
        args_dict.pop("kd_alpha")
        args_dict.pop("distill_temperature")
        args_dict.pop("distill_strategy")
        args_dict.pop("server_public_data_local_epoch")
        args_dict.pop("client_public_data_local_epoch")
        args_dict.pop("client_priv_data_local_epoch")
        args_dict.pop("global_epochs")

        args_dict.pop("skip_align", False)
        args_dict.pop("token_align_strategy")
        args_dict.pop("vocab_mapping_paths", None)
        args_dict.pop("vocab_size", None)

        args_dict.pop("post_fedavg")

        args_dict.pop("llm_training", True)
        args_dict.pop("use_mmlcc_aggregation", False)
        args_dict.pop("mmlcc_probability_epsilon", 1e-12)
        args_dict.pop("mmlcc_num_blocks", 1)
        args_dict.pop("mmlcc_privacy_guarantee", 1)
        args_dict.pop("mmlcc_beta_radius", 1.15)
        args_dict.pop("mmlcc_noise_sigma", 1.0)
        args_dict.pop("mmlcc_noise_clip_theta", 6.0)
        args_dict.pop("mmlcc_seed", 42)

        return args_dict

    def to_dict_with_client_priv_training_args(self):
        args_dict = self.to_dict_without_extra_args()

        args_dict["num_train_epochs"] = self.client_priv_data_local_epoch

        return args_dict

    def to_dict_with_client_kd_args(self):
        args_dict = self.to_dict_without_extra_args()

        args_dict["num_train_epochs"] = self.client_public_data_local_epoch

        return args_dict

    def to_dict_with_server_kd_args(self):
        args_dict = self.to_dict_without_extra_args()
        args_dict["num_train_epochs"] = self.server_public_data_local_epoch

        return args_dict


class FedMKTBase(object):
    def __init__(self, *args, **kwargs):
        self.model = None
        self.save_trainable_weights_only = None

    def save_model(
        self,
        output_dir: Optional[str] = None,
        state_dict=None
    ):
        if not self.save_trainable_weights_only:
            torch.save(self.model.state_dict(), output_dir + '/pytorch_model.bin')
        else:
            model = unwrap_model(self.model)

            if hasattr(model, "save_trainable"):
                model.save_trainable(output_dir)
            else:
                state_dict = {
                    k: p.to("cpu") for k,
                                       p in model.named_parameters() if p.requires_grad
                }

                torch.save(state_dict, output_dir + '/pytorch_model.bin')


class FedMKTSLM(FedMKTBase):
    def __init__(
        self,
        ctx: Context,
        model: torch.nn.Module,
        training_args: FedMKTTrainingArguments,
        fed_args: FedArguments = None,
        priv_train_set=None,
        pub_train_set=None,
        val_set: Dataset = None,
        priv_optimizer: torch.optim.Optimizer = None,
        pub_optimizer: torch.optim.Optimizer = None,
        priv_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        pub_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        save_trainable_weights_only: bool = False,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        llm_tokenizer=None,
        llm_to_slm_vocab_mapping=None,
    ):
        super(FedMKTSLM, self).__init__()
        self.ctx = ctx
        self.training_args = training_args
        self.fed_args = fed_args
        self.model = model
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.callbacks = callbacks
        self.compute_metrics = compute_metrics
        self.save_trainable_weights_only = save_trainable_weights_only
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        self.priv_data_collator = data_collator
        self.priv_optimizer = priv_optimizer
        self.pub_optimizer = pub_optimizer
        self.priv_scheduler = priv_scheduler
        self.pub_scheduler = pub_scheduler
        self.priv_train_set = priv_train_set
        self.pub_train_set = pub_train_set

        self.llm_tokenizer = llm_tokenizer
        self.llm_to_slm_vocab_mapping = llm_to_slm_vocab_mapping

        self.val_set = val_set

        self.aggregator = self._init_aggregator(ctx, fed_args)

        if not isinstance(self.pub_train_set, datasets.Dataset):
            self.pub_train_set = datasets.Dataset.from_list(list(self.pub_train_set))

    def train(self):
        global_epochs = self.training_args.global_epochs

        llm_pub_logits = None
        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            logger.info(f"begin {i}-th global kd process")
            self._fedmkt_current_round = i
            priv_data_training_args = self._get_priv_data_training_args()
            _prepare_dispatched_model_for_trainer(self.model)

            priv_trainer = Seq2SeqTrainer(
                model=self.model,
                data_collator=self.priv_data_collator,
                train_dataset=self.priv_train_set,
                args=priv_data_training_args,
                model_init=self.model_init if not i else None,
                compute_metrics=self.compute_metrics,
                callbacks=self.callbacks,
                optimizers=(self.priv_optimizer, self.priv_scheduler),
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
                **_tokenizer_init_kwargs(Seq2SeqTrainer, self.tokenizer),
            )
            _log_model_runtime_device(
                f"client_priv_trainer_round_{i}",
                priv_trainer.model,
                priv_data_training_args,
            )

            logger.info(f"begin {i}-th private data training process")
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
                    fn_kwargs={"model": self.model,
                               "training_args": self.training_args,
                               "data_collator": transformers.DataCollatorForSeq2Seq(self.tokenizer)}
                )

                if self.training_args.world_size > 1:
                    logger.info("sync slm_pub_logits")
                    sync_dataset(
                        slm_pub_logits, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
                    )

                if self.training_args.llm_training:
                    logger.debug(f"send {i}-th public logits to llm")
                    iter_ctx.arbiter.put("slm_pub_logits", slm_pub_logits.to_dict())

                if self.training_args.llm_training or not i:
                    llm_pub_logits = datasets.Dataset.from_dict(iter_ctx.arbiter.get("llm_pub_logits"))
                    if self.training_args.world_size > 1:
                        logger.info("sync llm_pub_logits")
                        sync_dataset(llm_pub_logits, self.training_args.local_rank,
                                     self.training_args.world_size, self.training_args.device)
            else:
                slm_pub_logits = sync_dataset(
                    None, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
                )

                if self.training_args.llm_training or not i:
                    llm_pub_logits = sync_dataset(None, self.training_args.local_rank,
                                                  self.training_args.world_size, self.training_args.device)

            logger.info(f"begin {i}-th token alignment process")
            aligned_dataset = token_align(
                base_model_logits_datasets=slm_pub_logits,
                blending_model_logits_dataset=llm_pub_logits,
                base_tokenizer=self.tokenizer,
                blending_tokenizer=self.llm_tokenizer,
                blending_to_base_mapping=self.llm_to_slm_vocab_mapping,
                blending_model_index=0,
                skip_align=self.training_args.skip_align,
                align_strategy=self.training_args.token_align_strategy
            )

            logger.info(f"begin {i}-th public logits kd process")
            fedmkt_trainer = self._init_trainer_for_distill(aligned_dataset)
            _log_model_runtime_device(
                f"client_kd_trainer_round_{i}",
                fedmkt_trainer.model,
                fedmkt_trainer.args,
            )
            fedmkt_trainer.train()
            self.model = unwrap_model(fedmkt_trainer.model)

            if self.training_args.post_fedavg and (i + 1) % self.fed_args.aggregate_freq == 0:
                self.aggregator.model_aggregation(iter_ctx, self.model)

            _log_round_arc_accuracy(self.model, self.tokenizer, i, "client")

    def _init_trainer_for_distill(self, train_set):
        public_data_training_args = self._get_pub_data_kd_training_args()
        fedmkt_trainer = FedMKTTrainer(
            model=self.model,
            args=public_data_training_args,
            train_dataset=train_set,
            eval_dataset=self.val_set,
            data_collator=DataCollatorForFedMKT(
                self.tokenizer,
                padding="max_length",
                max_length=max(len(d["input_ids"]) for d in train_set),
                blending_num=1,
                vocab_size=self.training_args.vocab_size,
                dtype=next(self.model.parameters()).dtype,
                distill_temperature=self.training_args.distill_temperature
            ),
            blending_num=1,
            lm_loss_weight=self.training_args.kd_alpha,
            distill_loss_type=self.training_args.distill_loss_type,
            distill_strategy=self.training_args.distill_strategy,
            loss_log_prefix="client",
            round_idx=getattr(self, "_fedmkt_current_round", None),
            **_tokenizer_init_kwargs(FedMKTTrainer, self.tokenizer),
        )

        return fedmkt_trainer

    def _get_priv_data_training_args(self):
        pre_args = self.training_args.to_dict_with_client_priv_training_args()
        post_args = Seq2SeqTrainingArguments(**pre_args)
        _force_training_args_device_from_env(post_args)

        return post_args

    def _get_pub_data_kd_training_args(self):
        pre_args = self.training_args.to_dict_with_client_kd_args()
        post_args = Seq2SeqTrainingArguments(**pre_args)
        _force_training_args_device_from_env(post_args)

        return post_args

    def _init_aggregator(self, ctx: Context, fed_args: FedArguments):
        if not self.training_args.post_fedavg:
            return None

        aggregate_type = "weighted_mean"
        aggregator_name = "fedavg"
        aggregator = fed_args.aggregator
        return AggregatorClientWrapper(
            ctx, aggregate_type, aggregator_name, aggregator,
            sample_num=len(self.pub_train_set), args=self.training_args
        )


class FedMKTLLM(FedMKTBase):
    def __init__(
        self,
        ctx: Context,
        model: torch.nn.Module,
        training_args: FedMKTTrainingArguments,
        fed_args: FedArguments = None,
        train_set=None,
        val_set: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        save_trainable_weights_only: bool = False,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        slm_tokenizers: List = None,
        slm_to_llm_vocab_mappings: List[Dict] = None,
    ):
        super(FedMKTLLM, self).__init__()
        self.ctx = ctx
        self.model = model
        self.training_args = training_args
        self.fed_args = fed_args
        self.train_set = train_set
        self.val_set = val_set
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.save_trainable_weights_only = save_trainable_weights_only
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.slm_tokenizers = slm_tokenizers
        self.slm_to_llm_vocab_mappings = slm_to_llm_vocab_mappings

        self.aggregator = self._init_aggregator(ctx)

        if not isinstance(self.train_set, datasets.Dataset):
            self.train_set = datasets.Dataset.from_list(list(self.train_set))

    def _init_aggregator(self, ctx: Context):
        if not self.training_args.post_fedavg:
            return None
        return AggregatorServerWrapper(ctx)

    def generate_pub_data_logits(self, first_epoch=False):
        fn_kwargs = {"model": self.model,
                     "training_args": self.training_args,
                     "data_collator": transformers.DataCollatorForSeq2Seq(self.tokenizer)}
        #把样本打包成 batch
        if first_epoch:
            _move_model_to_training_device(self.model, self.training_args.device)
        #调用 HuggingFace datasets.map() 方法，批量跑 generate_pub_data_logits 函数
        #此处的generate_pub_data_logits方法参考generate_logits_utils文件
        return self.train_set.map(
            generate_pub_data_logits,
            batched=True,
            batch_size=self.training_args.per_device_train_batch_size,
            num_proc=None,
            load_from_cache_file=False,
            fn_kwargs=fn_kwargs
        )

    def on_epoch_begin(self, iter_ctx, epoch_idx, previous_pub_dataset):
        logger.info(f"on {epoch_idx}-epoch begin")
        if not self.training_args.llm_training:
            return
        #第一轮才生成新的logits，单卡时自己算，多卡时只允许 rank=0 算，多卡时要从 rank=0 同步避免重复计算
        if previous_pub_dataset is None:
            #单机单卡 (world_size ≤ 1)，或者 多卡训练里的 rank=0 进程
            if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
                #如果是 第一个 epoch（epoch_idx=0），则设置 first_epoch=True，表示需要完整生成一次公共数据的 logits。
                #否则就是后续 epoch，可以做增量更新或者直接使用缓存。
                llm_pub_logits = self.generate_pub_data_logits(first_epoch=True if not epoch_idx else False)
                if self.training_args.world_size > 1:
                    # 把 rank=0 进程生成的 llm_pub_logits 同步到所有其他进程，保证 每个 GPU 上拿到完全一致的公共数据 logits，
                    # 否则多卡训练时各自算的 logits 会不一致。HuggingFace / PyTorch DDP 的默认逻辑
                    # 当你调用 .map() 或者 DataLoader 时，每个 rank（即每张卡上的进程）都会各自执行一份同样的 Python 代码
                    sync_dataset(llm_pub_logits, self.training_args.local_rank,
                                 self.training_args.world_size, self.training_args.device)
            else:
                #在多卡训练（分布式环境） 下，非rank=0 的进程（比如 GPU1、GPU2、GPU3）调用的逻辑，用来从 rank=0 同步拿到公共数据logits
                llm_pub_logits = sync_dataset(None, self.training_args.local_rank,
                                              self.training_args.world_size, self.training_args.device)
        else:
            llm_pub_logits = previous_pub_dataset

        #收集各个小模型（SLM）在公共数据集上的 logits
        slm_pub_logits_list = list()
        if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
            slm_pub_logits_list.append(datasets.Dataset.from_dict(iter_ctx.guest.get('slm_pub_logits')))
            if any(p.role == 'host' for p in self.ctx.parties):
                #rank=0 会汇总 guest + host 所有 SLM 的 logits
                slm_pub_logits_list.extend(
                    datasets.Dataset.from_dict(client_logits) for client_logits in iter_ctx.hosts.get("slm_pub_logits")
                )
            #多卡时 rank=0 还要负责把收集好的 每一个 SLM 的 logits dataset 广播给其他 rank
            if self.training_args.world_size > 1:
                logger.info("sync dataset to other rank")
                for slm_pub_logits in slm_pub_logits_list:
                    sync_dataset(slm_pub_logits, self.training_args.local_rank,
                                 self.training_args.world_size, self.training_args.device)
                    logger.info("end to sync")
        #多卡里的 非 rank=0 自己不生成 logits，直接等 rank=0 广播
        else:
            logger.info("sync dataset from rank 0")
            for _ in range(len(self.slm_tokenizers)):
                slm_pub_logits_list.append(
                    sync_dataset(None, self.training_args.local_rank,
                                 self.training_args.world_size, self.training_args.device)
                )
            logger.info("end to sync dataset from rank 0")
        #把 LLM 的公共 logits 和每个 SLM 的公共 logits 对齐（token-level alignment），得到一个最终的统一对齐数据集。
        aligned_dataset = llm_pub_logits
        for idx, slm_pub_logits in enumerate(slm_pub_logits_list):
            aligned_dataset = token_align(
                base_model_logits_datasets=aligned_dataset,
                blending_model_logits_dataset=slm_pub_logits,
                base_tokenizer=self.tokenizer,
                blending_tokenizer=self.slm_tokenizers[idx],
                blending_to_base_mapping=self.slm_to_llm_vocab_mappings[idx],
                blending_model_index=idx,
                skip_align=self.training_args.skip_align,
                align_strategy=self.training_args.token_align_strategy
            )
        #skip_align 如果为 True，就跳过对齐（直接拼接，不做 token-level 转换） 调试用的
        #greedy matching（贪心对齐）subword merge（子词合并）embedding similarity（用向量相似度对齐）

        if self.training_args.use_mmlcc_aggregation:
            aligned_dataset, mmlcc_metrics = aggregate_aligned_slm_teachers_dataset(
                aligned_dataset=aligned_dataset,
                blending_num=len(slm_pub_logits_list),
                distill_temperature=self.training_args.distill_temperature,
                probability_epsilon=self.training_args.mmlcc_probability_epsilon,
                num_blocks=self.training_args.mmlcc_num_blocks,
                privacy_guarantee=self.training_args.mmlcc_privacy_guarantee,
                beta_radius=self.training_args.mmlcc_beta_radius,
                noise_sigma=self.training_args.mmlcc_noise_sigma,
                noise_clip_theta=self.training_args.mmlcc_noise_clip_theta,
                seed=self.training_args.mmlcc_seed + epoch_idx,
            )
            print(
                "[MMLCC] "
                f"round={epoch_idx} "
                f"K={mmlcc_metrics['num_blocks']} "
                f"T={mmlcc_metrics['privacy_guarantee']} "
                f"positions={mmlcc_metrics['positions']} "
                f"relative_error={mmlcc_metrics['relative_error']:.6e}",
                flush=True,
            )
            try:
                import wandb
                log_metrics = {
                    "server/fedmkt_round": int(epoch_idx),
                    "server/mmlcc_relative_error": mmlcc_metrics["relative_error"],
                    "server/mmlcc_positions": mmlcc_metrics["positions"],
                    "server/mmlcc_num_blocks_K": mmlcc_metrics["num_blocks"],
                    "server/mmlcc_privacy_guarantee_T": mmlcc_metrics["privacy_guarantee"],
                }
                if _wandb_is_active(wandb):
                    wandb.log(log_metrics)
                _swanlab_log(log_metrics)
            except ImportError:
                _swanlab_log(
                    {
                        "server/fedmkt_round": int(epoch_idx),
                        "server/mmlcc_relative_error": mmlcc_metrics["relative_error"],
                        "server/mmlcc_positions": mmlcc_metrics["positions"],
                        "server/mmlcc_num_blocks_K": mmlcc_metrics["num_blocks"],
                        "server/mmlcc_privacy_guarantee_T": mmlcc_metrics["privacy_guarantee"],
                    }
                )

        return aligned_dataset

    def on_epoch_end(self, iter_ctx, epoch_idx):
        logger.info(f"on {epoch_idx}-epoch end")
        if not self.training_args.llm_training and epoch_idx > 1:
            return

        llm_pub_logits = self.generate_pub_data_logits(first_epoch=True if not self.training_args.llm_training else False)

        if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
            iter_ctx.guest.put("llm_pub_logits", llm_pub_logits.to_dict())
            if len(self.slm_tokenizers) > 1:
                iter_ctx.hosts.put("llm_pub_logits", llm_pub_logits.to_dict())

            if self.training_args.post_fedavg and (epoch_idx + 1) % self.fed_args.aggregate_freq == 0:
                self.aggregator.model_aggregation(iter_ctx)

            if self.training_args.world_size > 1:
                sync_dataset(
                    llm_pub_logits, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
                )
        else:
            llm_pub_logits = sync_dataset(
                None, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
            )

        return llm_pub_logits

    def _get_pub_data_kd_training_args(self):
        pre_args = self.training_args.to_dict_with_server_kd_args()
        post_args = Seq2SeqTrainingArguments(**pre_args)
        _force_training_args_device_from_env(post_args)

        return post_args

    def train(self):
        global_epochs = self.training_args.global_epochs
        previous_pub_logits = None

        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            logger.info(f"begin {i}-th global kd process")
            self._fedmkt_current_round = i

            aligend_train_set = self.on_epoch_begin(iter_ctx, i, previous_pub_logits)
            if self.training_args.llm_training:
                public_data_training_args = self._get_pub_data_kd_training_args()
                blending_num = 1 if self.training_args.use_mmlcc_aggregation else len(self.slm_tokenizers)
                fedmkt_trainer = FedMKTTrainer(
                    model=self.model,
                    args=public_data_training_args,
                    train_dataset=aligend_train_set,
                    eval_dataset=self.val_set,
                    data_collator=DataCollatorForFedMKT(
                        self.tokenizer,
                        padding="max_length",
                        max_length=max(len(d["input_ids"]) for d in aligend_train_set),
                        blending_num=blending_num,
                        vocab_size=self.training_args.vocab_size,
                        dtype=next(self.model.parameters()).dtype,
                        distill_temperature=self.training_args.distill_temperature
                    ),
                    blending_num=blending_num,
                    lm_loss_weight=self.training_args.kd_alpha,
                    distill_loss_type=self.training_args.distill_loss_type,
                    distill_strategy=self.training_args.distill_strategy,
                    loss_log_prefix="server",
                    round_idx=i,
                    **_tokenizer_init_kwargs(FedMKTTrainer, self.tokenizer),
                )

                fedmkt_trainer.train()
                self.model = unwrap_model(fedmkt_trainer.model)
                _log_round_arc_accuracy(self.model, self.tokenizer, i, "server")
            #在 epoch 结束时，生成新的 LLM 公共数据 logits，并负责 存储 / 广播 / 聚合
            previous_pub_logits = self.on_epoch_end(iter_ctx, i)
