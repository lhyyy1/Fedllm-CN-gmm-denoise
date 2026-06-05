# 项目目录说明

本文档按目录说明项目文件职责。当前项目主线围绕 `doc/tutorial/fedmkt/test.py` 展开。

## 根目录

```text
E:\Codes\test
├── main.py
├── README.md
├── PROJECT_DIRECTORY.md
├── doc/
├── lcc_fl/
├── models/
└── python/
```

### `main.py`

一个独立的小脚本，用于加载两个 LLaMA tokenizer 并比较词表大小和词表内容：

```python
tok7 = AutoTokenizer.from_pretrained("./models/llama-2-7b-hf")
tok13 = AutoTokenizer.from_pretrained("/home/cmcc/went/models/Llama-2-13b-hf")
```

它不是当前 FedMKT 主流程入口。 只用来测试

### `README.md`

项目说明和核心操作流程。

### `PROJECT_DIRECTORY.md`

项目目录职责说明。

## `doc/tutorial/fedmkt`

FedMKT 示例、配置、数据准备和核心运行入口所在目录。

```text
doc/tutorial/fedmkt
├── test.py
├── fedmkt_4_slms.py
├── prepare_fedmkt_data.py
├── fedmkt.ipynb
├── configs/
│   └── default.yaml
├── mapping/
│   ├── llama_small_to_llama.json
│   ├── llama_to_bloom.json
│   ├── llama_to_gpt2.json
│   └── llama_to_opt.json
└── __pycache__/
```

### `test.py`

当前项目核心运行文件。

主要职责：

- 加载实验配置。
- 解析任务、数据集、数据目录、模型路径、LoRA、训练参数、运行设备、日志平台和 MMLCC 参数。
- 启动 FATE 多进程训练。
- 根据角色分别调用 `train_llm` 或 `train_slm`。
- 为 LLM/SLM 构造 LoRA 模型、数据集、tokenizer 和训练参数。
- 调用 `FedMKTLLM`、`FedMKTSLM` 执行多轮联邦蒸馏训练。
- 训练后调用 `evaluate_task_accuracy` 做任务评估。
- 保存 LLM 和 SLM 的可训练权重。

### `fedmkt_4_slms.py`

较早版本的 4 个 SLM FedMKT 示例脚本。它同样包含 `train_llm`、`train_slm` 和 `run`，但配置能力、评估逻辑、日志控制和 MMLCC 支持都不如 `test.py` 完整。

### `prepare_fedmkt_data.py`

FedMKT 数据准备脚本。

主要职责：

- 支持 `arc_challenge`、`arc_easy`、`boolq`、`commonsenseqa`、`rte` 等数据集别名。
- 读取 `configs/default.yaml` 或命令行参数。
- 从 HuggingFace datasets 加载原始数据。
- 将训练集洗牌后切分为 4 个客户端私有数据和 1 个公共数据：
  - `client_0`
  - `client_1`
  - `client_2`
  - `client_3`
  - `common`
- 保留 `validation` / `test` split。
- 使用 `save_to_disk` 保存给 `test.py` 读取。

### `configs/default.yaml`

FedMKT 默认实验配置。

主要配置块：

- `data`：任务、数据集、数据目录、评估 split、公共数据采样范围。
- `data_prepare`：客户端数量、源 split、随机种子。
- `paths`：LLM/SLM 模型路径、映射目录、映射文件名、训练输出目录、模型保存目录。
- `lora`：LoRA rank、alpha、dropout、target modules。
- `training`：轮数、batch、学习率、蒸馏参数、优化器、dtype、device map。
- `runtime`：arbiter/guest/host 角色对应的 GPU。
- `mmlcc`：MMLCC 风格聚合参数。
- `wandb`：wandb 日志参数。
- `swanlab`：SwanLab 日志参数。

### `mapping`

词表映射文件目录，用于不同模型 tokenizer 之间的 token 对齐。

当前存在：

```text
llama_small_to_llama.json
llama_to_bloom.json
llama_to_gpt2.json
llama_to_opt.json
```

`test.py` 默认配置还会引用若干 SLM 到 LLM 方向的映射文件，实际运行前需要确保文件存在或修改配置。

## `python`

本地 Python 包源码目录，包名为 `fate_llm`。

```text
python
├── MANIFEST.in
├── requirements.txt
├── setup.py
└── fate_llm/
```

### `python/setup.py`

定义 `fate_llm` 包、依赖和命令行入口：

```text
fate_llm = fate_llm.evaluate.scripts.fate_llm_cli:fate_llm_cli
```

### `python/requirements.txt`

项目 Python 依赖清单。本文档不展开环境配置步骤。

## `python/fate_llm`

FATE-LLM 主要源码。

```text
python/fate_llm
├── algo/
├── data/
├── dataset/
├── evaluate/
├── inference/
├── model_zoo/
├── runner/
├── trainer/
└── __init__.py
```

### `algo`

联邦学习和隐私相关算法实现。

```text
algo
├── dp/
├── fdkt/
├── fedavg/
├── fedkseed/
├── fedmkt/
├── inferdpt/
├── offsite_tuning/
└── pdss/
```

当前 `test.py` 主要使用 `algo/fedmkt`。

### `algo/fedmkt`

FedMKT 核心实现。

```text
fedmkt
├── fedmkt.py
├── fedmkt_data_collator.py
├── fedmkt_trainer.py
├── mmlcc.py
├── token_alignment/
└── utils/
```

核心文件职责：

- `fedmkt.py`：定义 `FedMKTTrainingArguments`、`FedMKTLLM`、`FedMKTSLM`，负责 LLM/SLM 训练循环、logits 交换、token 对齐和可选聚合。
- `fedmkt_trainer.py`：继承 HuggingFace `Seq2SeqTrainer`，重写 `compute_loss`，将监督损失和蒸馏损失按 `kd_alpha` 混合。
- `fedmkt_data_collator.py`：为 FedMKT 蒸馏训练整理 batch，包括 teacher 分布、metric、attention mask 等。
- `mmlcc.py`：对多个对齐后的 SLM teacher 分布执行 MMLCC 风格聚合。
- `token_alignment/`：不同 tokenizer 之间的 vocab mapping 和 token-level alignment。
- `utils/`：logits 生成、数据同步、tokenizer 工具和变量名定义。

### `dataset`

数据集封装和预处理。

`test.py` 主要使用：

```text
python/fate_llm/dataset/qa_dataset.py
```

`qa_dataset.py` 提供：

- ARC、BoolQ、CommonsenseQA、RTE 等任务模板。
- `tokenize_qa_dataset(...)`：将原始样本转成 causal LM 训练格式。
- `QaDataset`：FATE 数据集封装，负责 `load_from_disk`、tokenize、select 子集。

### `model_zoo`

模型封装目录。

```text
model_zoo
├── hf_model.py
├── embedding_transformer/
├── offsite_tuning/
└── pellm/
```

`test.py` 主要使用 `model_zoo/pellm` 中的参数高效 LLM 包装：

- `llama.py`：LLaMA causal LM 封装。
- `gpt2.py`：GPT2 causal LM 封装。
- `opt.py`：OPT causal LM 封装。
- `bloom.py`：Bloom causal LM 封装。
- `parameter_efficient_llm.py`：通用 PEFT/LoRA 基类。

### `trainer`

训练器基础封装。

`FedMKTTrainingArguments` 继承这里的：

```text
python/fate_llm/trainer/seq2seq_trainer.py
```

### `evaluate`

评估工具和 CLI。

与 `test.py` 相关的是：

```text
python/fate_llm/evaluate/arc_eval.py
```

此外 `test.py` 自身也实现了 ARC、RTE、BoolQ、CommonsenseQA 的评估函数。

### `data`

tokenizer 和 data collator 相关封装。

`test.py` 使用：

```text
python/fate_llm/data/tokenizers/cust_tokenizer.py
```

用于根据本地模型路径创建 tokenizer。

### `runner`

不同算法的运行器封装，包括：

- `fedmkt_runner.py`
- `fedkseed_runner.py`
- `fdkt_runner.py`
- `pdss_runner.py`
- `offsite_tuning_runner.py`

当前主流程直接使用 `doc/tutorial/fedmkt/test.py`，没有通过 runner 启动。

### `inference`

推理接口封装，包括 HuggingFace、本地 API、vLLM 等推理适配。

## `models`

本地模型目录。

当前扫描到：

```text
models
├── llama-2-7b-hf/
└── opt-1.3b/
```

`configs/default.yaml` 默认还引用了其他模型路径，例如 `gpt2-xl`、`Sheared-LLaMA-1.3B`、`bloom-1b1` 和 `/home/cmcc/went/models/Llama-2-13b-hf`。运行前需要让配置与实际模型路径一致。

## `lcc_fl`

独立的 LCC-FL 相关子项目，内部有自己的 `.git`、`README.md`、`requirements.txt` 和 `lcc/` 包。

```text
lcc_fl
├── README.md
├── requirements.txt
├── lcc/
│   ├── lcc.py
│   ├── oracles.py
│   └── __init__.py
├── lcc_linreg.ipynb
└── LCC.drawio
```

它不是 `test.py` 的直接依赖，但可能与 MMLCC/编码聚合思路有关。

## IDE 与缓存目录

以下目录属于工具或运行缓存，不属于核心业务代码：

```text
.idea/
**/__pycache__/
models/**/.cache/
lcc_fl/.git/
```

整理项目或提交代码时，通常不需要关注这些目录。
