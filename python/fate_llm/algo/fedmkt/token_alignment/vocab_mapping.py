#
# NOTE: The find_best_mapping function is copied from FuseAI/FuseLLM
# Copyright FuseAI/FuseLLM
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
import json
import editdistance
import tqdm
import multiprocessing
import logging

from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
from fate_llm.algo.fedmkt.token_alignment.spectal_token_mapping import get_special_token_marker

logger = logging.getLogger(__name__)


def find_best_mapping(x, base_tokens, blending_model_special_token, base_model_special_token, best_one=True):
    """code refer to https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/vocab_mapping.py#L82"""
    tmp_x = x.replace(blending_model_special_token, base_model_special_token)
    if tmp_x in base_tokens:
        return tmp_x, tmp_x
    else:
        if best_one:
            return tmp_x, min([(y, editdistance.eval(tmp_x, y)) for y in base_tokens], key=lambda d: d[1])[0]
        else:
            token_and_distance = [(y, editdistance.eval(tmp_x, y)) for y in base_tokens]
            min_distance = min(item[1] for item in token_and_distance)
            shortest_distance_tokens = [item[0] for item in token_and_distance if item[1] == min_distance]
            return tmp_x, shortest_distance_tokens

#在两个模型的词表之间建立最佳的 token 对应关系，并保存成 JSON 文件
def get_vocab_mappings(model_name_or_path, candidate_model_name_or_path, vocab_mapping_save_path, num_processors=8):
    #加载两个模型的 tokenizer
    ori_tokenizer = get_tokenizer(model_name_or_path)
    candidate_tokenizer = get_tokenizer(candidate_model_name_or_path)
    #取出特殊字符 这些特殊符号需要单独对齐，不是普通词汇
    ori_special_tok = get_special_token_marker(ori_tokenizer)
    candidate_special_tok = get_special_token_marker(candidate_tokenizer)
    #获取候选模型的所有 token（词表）
    candidate_tokens = list(candidate_tokenizer.get_vocab().keys())
    #找源模型的 token tok在候选模型词表 candidate_tokens里的最佳匹配
    with multiprocessing.Pool(num_processors) as process_pool:
        func_args = [(tok, candidate_tokens, ori_special_tok, candidate_special_tok) for tok in ori_tokenizer.get_vocab()]

        vocab_mappings = dict(tqdm.tqdm(process_pool.starmap(find_best_mapping, func_args)),
                              total=len(ori_tokenizer.get_vocab()))

    with open(vocab_mapping_save_path, "w") as fout:
        json.dump(vocab_mappings, fout)

    return vocab_mappings

    """
    eg.
    {
    "dog": "dog",
    "cat": "cat",
    "New_York": "New York",
    "<unk>": "<unk>"
}
    """
