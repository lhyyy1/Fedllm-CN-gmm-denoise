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
import transformers


SENTENCEPIECE_SPACE = "\u2581"
BYTE_BPE_SPACE = "\u0120"

TOKENIZER_TO_SPECIAL_TOKEN = {}


def _register_tokenizer(tokenizer_name, marker):
    try:
        tokenizer_cls = getattr(transformers, tokenizer_name)
    except (AttributeError, ImportError):
        return
    TOKENIZER_TO_SPECIAL_TOKEN[tokenizer_cls] = marker


for _tokenizer_name in ("LlamaTokenizer", "LlamaTokenizerFast", "GemmaTokenizer", "GemmaTokenizerFast"):
    _register_tokenizer(_tokenizer_name, SENTENCEPIECE_SPACE)

for _tokenizer_name in ("GPTNeoXTokenizerFast", "GPT2TokenizerFast", "GPT2Tokenizer", "BloomTokenizerFast"):
    _register_tokenizer(_tokenizer_name, BYTE_BPE_SPACE)


def get_special_token_marker(tokenizer):
    marker = TOKENIZER_TO_SPECIAL_TOKEN.get(tokenizer.__class__)
    if marker is not None:
        return marker

    class_name = tokenizer.__class__.__name__.lower()
    if any(name in class_name for name in ("llama", "gemma", "sentencepiece")):
        return SENTENCEPIECE_SPACE
    if any(name in class_name for name in ("gpt2", "gptneox", "bloom", "falcon", "mpt")):
        return BYTE_BPE_SPACE

    vocab = tokenizer.get_vocab()
    for token in vocab:
        if token.startswith(SENTENCEPIECE_SPACE):
            return SENTENCEPIECE_SPACE
        if token.startswith(BYTE_BPE_SPACE):
            return BYTE_BPE_SPACE

    raise KeyError(
        f"unsupported tokenizer class {tokenizer.__class__.__name__}; "
        "cannot infer token space marker for vocab alignment"
    )
