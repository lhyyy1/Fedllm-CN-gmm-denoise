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
import peft
import torch
from collections.abc import Mapping
from peft import PeftModel, TaskType
from transformers import AutoConfig
from transformers import AutoModel
from transformers.configuration_utils import PretrainedConfig
import logging


logger = logging.getLogger(__name__)


AVAILABLE_PEFT_CONFIG = list(
    filter(
        lambda peft_type: peft_type.endswith("Config"), dir(peft)
    )
)


class PELLM(torch.nn.Module):

    config_class: PretrainedConfig = None
    model_loader = None

    def __init__(self,
                 config: dict = None,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config=None,
                 torch_dtype: str = None,
                 trust_remote_code: bool = False,
                 model_load_kwargs: dict = None,
                 **kwargs
                 ) -> None:

        super().__init__()
        self._pe_lm: PeftModel = None
        self.config = config
        self.config_path = pretrained_path
        self.peft_type = peft_type
        self.peft_config = peft_config
        self.torch_dtype = None if not torch_dtype else getattr(torch, torch_dtype)
        self.trust_remote_code = trust_remote_code
        self.model_load_kwargs = model_load_kwargs or {}

        assert self.config_path is not None or self.config is not None, \
            "At least one of config_path and config must be set."
        self._init_pelm(**kwargs)

    def _init_pelm(self, **kwargs):
        self.init_lm_with_peft(**kwargs)
        self.model_summary()

    def init_lm_with_peft(self, **kwargs):
        self.init_config(**kwargs)
        self.init_base_lm(**self.model_load_kwargs)
        self.add_peft()

    def init_config(self, **kwargs):
        if self.config_path is not None:
            self.config = AutoConfig.from_pretrained(self.config_path, trust_remote_code=self.trust_remote_code)
        elif self.config is not None and self.config_class is not None:
            self.config = self.config_class().from_dict(self.config)
        else:
            raise ValueError(
                'config_path to pretrained model folder and model config dict cannot be None at the same time, '
                'you need to specify one of them')

        if kwargs:
            self.config.update(kwargs)

    def init_base_lm(self, **kwargs):
        model_loader = self.model_loader if self.model_loader is not None else AutoModel
        if self.config is not None:
            self._pe_lm = model_loader.from_pretrained(
                self.config_path, config=self.config,
                torch_dtype=self.torch_dtype, **kwargs,
                trust_remote_code=self.trust_remote_code
            )
        elif self.config_path is not None:
            self._pe_lm = model_loader.from_pretrained(
                self.config_path, torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code, **kwargs)
        else:
            raise ValueError(
                'config_path to pretrained model folder cannot be None')

    def add_peft(self):
        assert self.peft_type in AVAILABLE_PEFT_CONFIG, 'peft name {} not in available config {}'.format(
            self.peft_type, AVAILABLE_PEFT_CONFIG)

        if self.peft_config is None:
            peft_config = getattr(peft, self.peft_type)()
        elif isinstance(self.peft_config, dict):
            peft_config = getattr(peft, self.peft_type)(**self.peft_config)
        else:
            raise ValueError(f"Can not parse peft_config of {type(self.peft_config)}")

        self._pe_lm = peft.get_peft_model(self._pe_lm, peft_config)
        self.peft_config = peft_config

    def model_summary(self):
        if hasattr(self._pe_lm, "print_trainable_parameters"):
            summary = self._pe_lm.print_trainable_parameters()
            logger.debug(f'PELLM model summary: \n{summary}')

    def _iter_wrapped_models(self):
        seen = set()
        candidates = [self._pe_lm]
        while candidates:
            candidate = candidates.pop()
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            yield candidate

            for attr_name in ("module", "base_model", "model"):
                child = getattr(candidate, attr_name, None)
                if child is not None and child is not candidate:
                    candidates.append(child)

    @property
    def hf_device_map(self):
        for model in self._iter_wrapped_models():
            device_map = getattr(model, "hf_device_map", None)
            if device_map is not None:
                return device_map
        raise AttributeError("hf_device_map")

    def _has_meta_parameters(self):
        return any(param.is_meta for param in self.parameters())

    def _has_hf_device_map(self):
        try:
            self.hf_device_map
        except AttributeError:
            return False
        return True

    def _is_dispatched_or_lazy_loaded(self):
        return self._has_hf_device_map() or self._has_meta_parameters()

    def to(self, *args, **kwargs):
        if self._is_dispatched_or_lazy_loaded():
            logger.info("skip PELLM.to(); model is managed by hf device_map/lazy loading")
            return self
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        if self._is_dispatched_or_lazy_loaded():
            logger.info("skip PELLM.cuda(); model is managed by hf device_map/lazy loading")
            return self
        return super().cuda(device)

    def cpu(self):
        if self._is_dispatched_or_lazy_loaded():
            logger.info("skip PELLM.cpu(); model is managed by hf device_map/lazy loading")
            return self
        return super().cpu()

    def forward(self, *args, **kwargs):
        forward_ret = self._pe_lm.forward(*args, **kwargs)

        if self.peft_config is None or self.peft_config.task_type != TaskType.SEQ_CLS:
            return forward_ret
        else:
            return forward_ret.logits

    def save_trainable(self, output_path):
        self._pe_lm.save_pretrained(output_path)


class AutoPELLM(PELLM):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
