import json
import logging
from typing import Dict, List, Literal, Optional, Union

from fate.arch.dataframe import DataFrame
from fate.components.components.nn.loader import Loader
from fate.components.components.nn.nn_runner import (
    dir_warning,
    load_model_dict_from_path,
    loader_load_from_conf,
)
from fate.components.components.nn.runner.homo_default_runner import DefaultRunner
from fate.ml.nn.homo.fedavg import FedAVGArguments
from transformers.trainer_utils import get_last_checkpoint

from fate_llm.algo.trust_align import (
    TrustedFedMKTLLM,
    TrustedFedMKTSLM,
    TrustedFedMKTTrainingArguments,
)

logger = logging.getLogger(__name__)


class TrustedFedMKTRunner(DefaultRunner):
    """Runner for trusted cloud-edge LLM/SLM co-training.

    Usage is identical to FedMKTRunner except algo="trusted_fedmkt" and the
    training_args_conf may include use_trust_align/trust_* options.
    """

    def __init__(
        self,
        algo: str = "trusted_fedmkt",
        model_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        fed_args_conf: Optional[Dict] = None,
        pub_dataset_conf: Optional[Dict] = None,
        priv_dataset_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        llm_tokenizer_conf: Optional[Dict] = None,
        slm_tokenizers_conf: List[Optional[Dict]] = None,
        llm_to_slm_vocab_mapping_path: str = None,
        slm_to_llm_vocab_mapping_paths: List[str] = None,
        task_type: Literal["causal_lm", "others"] = "causal_lm",
        save_trainable_weights_only: bool = False,
        pub_dataset_path: str = None,
    ) -> None:
        super().__init__()
        self.algo = algo
        self.model_conf = model_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.fed_args_conf = fed_args_conf
        self.pub_dataset_conf = pub_dataset_conf
        self.priv_dataset_conf = priv_dataset_conf
        self.data_collator_conf = data_collator_conf
        self.tokenizer_conf = tokenizer_conf
        self.llm_tokenizer_conf = llm_tokenizer_conf
        self.slm_tokenizers_conf = slm_tokenizers_conf
        self.llm_to_slm_vocab_mapping_path = llm_to_slm_vocab_mapping_path
        self.slm_to_llm_vocab_mapping_paths = slm_to_llm_vocab_mapping_paths
        self.task_type = task_type
        self.pub_dataset_path = pub_dataset_path
        self.save_trainable_weights_only = save_trainable_weights_only
        self.training_args = None
        if self.algo.lower() != "trusted_fedmkt":
            raise ValueError("algo should be trusted_fedmkt")
        if self.task_type not in ["causal_lm"]:
            raise ValueError("task_type should be causal_lm")

    def common_setup(self, saved_model=None, output_dir=None):
        ctx = self.get_context()
        output_dir = output_dir or "./"
        model = loader_load_from_conf(self.model_conf)
        if model is None:
            raise ValueError(f"model is None, cannot load model from conf {self.model_conf}")

        resume_path = None
        if saved_model is not None:
            model.load_state_dict(load_model_dict_from_path(saved_model))
            if get_last_checkpoint(saved_model) is not None:
                resume_path = saved_model

        if self.optimizer_conf:
            optimizer_loader = Loader.from_dict(self.optimizer_conf)
            optimizer_cls = optimizer_loader.load_item()
            optimizer = optimizer_cls(model.parameters(), **optimizer_loader.kwargs)
        else:
            optimizer = None

        tokenizer = loader_load_from_conf(self.tokenizer_conf)
        dir_warning(self.training_args_conf)
        training_args = TrustedFedMKTTrainingArguments(**self.training_args_conf)
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path
        self.training_args = training_args

        fed_args = FedAVGArguments(**self.fed_args_conf) if self.fed_args_conf is not None else None
        pub_dataset = loader_load_from_conf(self.pub_dataset_conf)
        pub_dataset.load(self.pub_dataset_path)
        return ctx, model, optimizer, tokenizer, training_args, fed_args, pub_dataset

    def llm_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        ctx, model, optimizer, tokenizer, training_args, fed_args, pub_dataset = self.common_setup(output_dir=output_dir, saved_model=saved_model)
        validate_dataset = None
        if validate_set is not None:
            validate_dataset = loader_load_from_conf(self.pub_dataset_conf)
            validate_dataset.load(validate_set)
        slm_tokenizers = [loader_load_from_conf(c) for c in self.slm_tokenizers_conf] if self.slm_tokenizers_conf else []
        slm_to_llm_vocab_mappings = []
        for path in self.slm_to_llm_vocab_mapping_paths:
            with open(path, "r") as fin:
                slm_to_llm_vocab_mappings.append(json.loads(fin.read()))
        return TrustedFedMKTLLM(
            ctx=ctx,
            model=model,
            training_args=training_args,
            fed_args=fed_args,
            train_set=pub_dataset,
            val_set=validate_dataset,
            tokenizer=tokenizer,
            slm_tokenizers=slm_tokenizers,
            slm_to_llm_vocab_mappings=slm_to_llm_vocab_mappings,
            save_trainable_weights_only=self.save_trainable_weights_only,
        )

    def slm_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        ctx, model, optimizer, tokenizer, training_args, fed_args, pub_dataset = self.common_setup(output_dir=output_dir, saved_model=saved_model)
        priv_dataset = loader_load_from_conf(self.priv_dataset_conf)
        priv_dataset.load(train_set)
        validate_dataset = None
        if validate_set is not None:
            validate_dataset = loader_load_from_conf(self.priv_dataset_conf)
            validate_dataset.load(validate_set)
        llm_tokenizer = loader_load_from_conf(self.llm_tokenizer_conf)
        with open(self.llm_to_slm_vocab_mapping_path, "r") as fin:
            vocab_mapping = json.loads(fin.read())
        priv_data_collator = loader_load_from_conf(self.data_collator_conf)
        return TrustedFedMKTSLM(
            ctx=ctx,
            model=model,
            training_args=training_args,
            fed_args=fed_args,
            pub_train_set=pub_dataset,
            priv_train_set=priv_dataset,
            val_set=validate_dataset,
            tokenizer=tokenizer,
            save_trainable_weights_only=self.save_trainable_weights_only,
            llm_tokenizer=llm_tokenizer,
            llm_to_slm_vocab_mapping=vocab_mapping,
            data_collator=priv_data_collator,
        )

    def train(
        self,
        train_data: Optional[Union[str, DataFrame]] = None,
        validate_data: Optional[Union[str, DataFrame]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ):
        if self.is_client():
            trainer = self.slm_setup(train_set=train_data, validate_set=validate_data, output_dir=output_dir, saved_model=saved_model_path)
        else:
            trainer = self.llm_setup(train_set=train_data, validate_set=validate_data, output_dir=output_dir, saved_model=saved_model_path)
        trainer.train()
        trainer.save_model(output_dir)
