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

__all__ = [
    "FedMKTSLM",
    "FedMKTLLM",
    "FedMKTTrainingArguments"
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from fate_llm.algo.fedmkt.fedmkt import (
        FedMKTTrainingArguments,
        FedMKTSLM,
        FedMKTLLM,
    )

    return {
        "FedMKTSLM": FedMKTSLM,
        "FedMKTLLM": FedMKTLLM,
        "FedMKTTrainingArguments": FedMKTTrainingArguments,
    }[name]
