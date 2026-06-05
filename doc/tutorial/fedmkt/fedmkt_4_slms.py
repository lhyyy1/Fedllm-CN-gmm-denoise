# fedmkt_4_slms.py

import os

from fate.arch import Context
from fate.arch.launchers.multiprocess_launcher import launch
import json
#这个在fate目录里 比较混乱 可设置单独文件夹最初用jupyter根目录执行错误
process_data_output_dir = "../../../../."
llm_pretrained_path = "../../../models/llama-2-7b-hf"
slm_0_pretrained_path = "../../../models/opt-1.3b"
slm_1_pretrained_path = "../../../models/gpt2-xl"
slm_2_pretrained_path = "../../../models/Sheared-LLaMA-1.3B"
slm_3_pretrained_path = "../../../models/bloom-1b1"
llm_slm_pairs = [
    (llm_pretrained_path, slm_0_pretrained_path),
    (llm_pretrained_path, slm_1_pretrained_path),
    (llm_pretrained_path, slm_2_pretrained_path),
    (llm_pretrained_path, slm_3_pretrained_path)
]

vocab_mapping_directory = "./mapping"

slm_to_llm_vocab_mapping_paths = ["opt_to_llama.json", "gpt2_to_llama.json", "llama_small_to_llama.json", "bloom_to_llama.json"]
llm_to_slm_vocab_mapping_paths = ["llama_to_opt.json", "llama_to_gpt2.json", "llama_to_llama_small", "llama_to_bloom.json"]

for idx in range(4):
    slm_to_llm_vocab_mapping_paths[idx] = vocab_mapping_directory + "/" + slm_to_llm_vocab_mapping_paths[idx]
    llm_to_slm_vocab_mapping_paths[idx] = vocab_mapping_directory + "/" + llm_to_slm_vocab_mapping_paths[idx]

slm_pretrained_paths = [slm_0_pretrained_path, slm_1_pretrained_path, slm_2_pretrained_path, slm_3_pretrained_path]
slm_lora_target_modules = [
    ["q_proj", "v_proj"],
    ["c_attn"],
    ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    ["query_key_value"]
]

global_epochs = 5
batch_size=4
llm_lr = 3e-5
slm_lrs = [3e-5, 3e-4, 3e-5, 3e-5, 3e-5]

#说明此处的
llm_model_saved_directory = "./models/fedmkt_4_slms_llm_model"
slm_models_saved_directory = [
    "./models/fedmkt_4_slms_slm_0",
    "./models/fedmkt_4_slms_slm_1",
    "./models/fedmkt_4_slms_slm_2",
    "./models/fedmkt_4_slms_slm_3"
]


def train_llm(ctx):
    import sys
    sys.path.insert(0, "./python")
    from peft import LoraConfig, TaskType
    from fate_llm.model_zoo.pellm.llama import LLaMa
    from fate_llm.algo.fedmkt import FedMKTTrainingArguments, FedMKTLLM
    from fate_llm.dataset.qa_dataset import QaDataset
    from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
    from transformers import AutoConfig

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )

    model = LLaMa(
        pretrained_path=llm_pretrained_path,
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype="float32"
    )

    pub_data = QaDataset(tokenizer_name_or_path=llm_pretrained_path,
                         dataset_name="arc_challenge",
                         data_part="common",
                         seq_max_len=512,
                         need_preprocess=True)
    pub_data.load(process_data_output_dir)

    training_args = FedMKTTrainingArguments(
        global_epochs=global_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=llm_lr,
        output_dir="../../../../.",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_ratio=0.008,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_grad_norm=1.0,
        use_cpu=False,
        vocab_size=AutoConfig.from_pretrained(llm_pretrained_path).vocab_size,
    )

    slm_to_llm_vocab_mapping = []
    for path in slm_to_llm_vocab_mapping_paths:
        with open(path, "r") as fin:
            vocab_mapping = json.loads(fin.read())
            slm_to_llm_vocab_mapping.append(vocab_mapping)

    slm_tokenizers = [get_tokenizer(slm_pretrained_path) for slm_pretrained_path in slm_pretrained_paths]

    tokenizer = get_tokenizer(llm_pretrained_path)
    trainer = FedMKTLLM(
        ctx=ctx,
        model=model,
        training_args=training_args,
        train_set=pub_data,
        tokenizer=tokenizer,
        slm_tokenizers=slm_tokenizers,
        slm_to_llm_vocab_mappings=slm_to_llm_vocab_mapping,
        save_trainable_weights_only=True,
    )

    trainer.train()
    trainer.save_model(llm_model_saved_directory)


def train_slm(ctx, slm_idx):
    import sys
    sys.path.insert(0, "./python")
    import transformers
    from peft import LoraConfig, TaskType
    from fate_llm.model_zoo.pellm.llama import LLaMa
    from fate_llm.model_zoo.pellm.gpt2 import GPT2CLM
    from fate_llm.model_zoo.pellm.opt import OPT
    from fate_llm.model_zoo.pellm.bloom import Bloom
    from fate_llm.algo.fedmkt import FedMKTTrainingArguments, FedMKTSLM
    from fate_llm.dataset.qa_dataset import QaDataset
    from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
    from transformers import AutoConfig

    slm_model_class = [
        OPT,
        GPT2CLM,
        LLaMa,
        Bloom
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=slm_lora_target_modules[slm_idx]
    )

    model = slm_model_class[slm_idx](
        pretrained_path=slm_pretrained_paths[slm_idx],
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype="float32"
    )

    priv_data = QaDataset(tokenizer_name_or_path=slm_pretrained_paths[slm_idx],
                          dataset_name="arc_challenge",
                          data_part=f"client_{slm_idx}",
                          seq_max_len=512,
                          need_preprocess=True)
    priv_data.load(process_data_output_dir)

    pub_data = QaDataset(tokenizer_name_or_path=slm_pretrained_paths[slm_idx],
                         dataset_name="arc_challenge",
                         data_part="common",
                         seq_max_len=512,
                         need_preprocess=True)
    pub_data.load(process_data_output_dir)

    training_args = FedMKTTrainingArguments(
        global_epochs=global_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=slm_lrs[slm_idx],
        output_dir="../../../../.",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_ratio=0.008,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_grad_norm=1.0,
        use_cpu=False,
        vocab_size=AutoConfig.from_pretrained(slm_pretrained_paths[slm_idx]).vocab_size,
    )

    tokenizer = get_tokenizer(slm_pretrained_paths[slm_idx])

    import json
    with open(llm_to_slm_vocab_mapping_paths[slm_idx], "r") as fin:
        vocab_mapping = json.loads(fin.read())

    trainer = FedMKTSLM(
        ctx=ctx,
        model=model,
        training_args=training_args,
        pub_train_set=pub_data,
        priv_train_set=priv_data,
        tokenizer=tokenizer,
        save_trainable_weights_only=True,
        llm_tokenizer=get_tokenizer(llm_pretrained_path),
        llm_to_slm_vocab_mapping=vocab_mapping,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer)
    )

    trainer.train()
    trainer.save_model(slm_models_saved_directory[slm_idx])


def run(ctx: Context):
    if ctx.is_on_arbiter:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        train_llm(ctx)
    elif ctx.is_on_guest:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        train_slm(ctx, slm_idx=0)
    else:
        if ctx.local.party[1] == "9999":
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            slm_idx = 1
        elif ctx.local.party[1] == "10000":
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            slm_idx = 2
        elif ctx.local.party[1] == "10001":
            os.environ["CUDA_VISIBLE_DEVICES"] = "4"
            slm_idx = 3
        else:
            raise ValueError(f"party_id={ctx.local.party[1]} is illegal")

        train_slm(ctx, slm_idx=slm_idx)


if __name__ == "__main__":
    launch(run)
