server03部署目录：
```text
home/cmcc/went/fate1
```

# FedMKT 多模型联邦知识迁移项目

当前项目的核心入口是：

```text
doc/tutorial/fedmkt/test.py
```

`test.py` 会通过 FATE库 的 `multiprocess_launcher.launch(run)` 启动多方进程：中心侧 LLM 作为 arbiter，多个 SLM 作为 guest/host。训练过程中，各 SLM 先在私有数据上训练，再在公共数据上生成 logits；中心 LLM 收集 SLM logits，完成 token 对齐、蒸馏训练，并将新的 LLM logits 广播回各 SLM，循环执行多轮全局训练。

## 一、研究任务与测试指标

本项目面向“多源跨域知识可信共享的体系架构”研究任务，当前 `test.py` 主要覆盖两个研究点：知识蒸馏驱动的联邦大小模型协同训练架构，以及拉格朗日冗余编码赋能的可信知识共享方法。

### 1. 知识蒸馏驱动的联邦大小模型协同训练架构

功能实现：

- 云侧大模型将通用知识传给端侧小模型。
- 端侧小模型将私域知识传给云侧大模型。
- 在数据及模型不出域的前提下，通过公共数据 logits、token alignment 和知识蒸馏实现大小模型双向知识迁移。

测试指标：

- 支持至少三种主流通用大模型作为中心侧 LLM：
  - `Llama-2-13b-hf`
  - `gemma-12b`
  - `Qwen2.5-14B`
- 支持四种私域 SLM：
  - `opt-1.3b`
  - `gpt2-xl`
  - `Sheared-LLaMA-1.3B`
  - `bloom-1b1`
- 支持多种问答/推理数据集：
  - `arc_challenge`
  - `boolq`
  - `arc_easy`
  - `cqa`
  - `rte`

### 2. 拉格朗日冗余编码赋能的可信知识共享方法

功能实现：

- 端侧将本地待传知识信息通过拉格朗日冗余编码划分成 encoded shares 后共享。
- 每个客户端获取来自其他客户端的 partial shares，并进行本地聚合后上传至服务器。
- 服务器通过插值算法重构完整知识信息，保证端云交互过程中的知识共享可控、可验证。
- 当前 FedMKT 主流程中的实现位于 `python/fate_llm/algo/fedmkt/mmlcc.py`，通过 `configs/default.yaml` 的 `mmlcc.use_aggregation` 控制是否启用。

测试指标：

- 关注编解码过程的相对误差。 https://swanlab.cn/@19854216519-/fedmkt/overview
- 当 `swanlab` 中的 `mmlcc_relative_error` 小于 `1e-10` 时，可认为编码聚合和原始直接聚合在数值上等价，误差可忽略。

## 二、项目结构

详细目录说明见 [PROJECT_DIRECTORY.md](PROJECT_DIRECTORY.md)。

核心目录如下：

```text
.
├── main.py
├── README.md
├── PROJECT_DIRECTORY.md
├── doc/
│   └── tutorial/
│       └── fedmkt/
│           ├── fedmkt_4_slms.py
│           ├── prepare_fedmkt_data.py
│           ├── configs/
│           │   └── default.yaml
│           └── mapping/
├── python/
│   └── fate_llm/
│       ├── algo/
│       ├── data/
│       ├── dataset/
│       ├── evaluate/
│       ├── inference/
│       ├── model_zoo/
│       ├── runner/
│       └── trainer/
├── models/
├── data/
└── lcc_fl/
```

## 三、核心运行入口

### `doc/tutorial/fedmkt/test.py`

这是当前主要运行文件，负责：

- 读取 `doc/tutorial/fedmkt/configs/default.yaml` 中的数据、模型、训练、运行设备、日志平台和 MMLCC 参数。
- 按 FATE 角色启动不同训练逻辑：
  - arbiter：训练中心 LLM。
  - guest：训练第 0 个 SLM。
  - host `9999`：训练第 1 个 SLM。
  - host `10000`：训练第 2 个 SLM。
  - host `10001`：训练第 3 个 SLM。 (guest和host只是name差异，本质上都是客户端)
- 加载公共数据 `common` 和客户端私有数据 `client_i`。
- 加载 LLM/SLM 词表映射文件，执行 logits 的 token alignment。
- 使用 LoRA 方式加载和训练 LLM/SLM。
- 每轮执行 FedMKT 蒸馏训练。
- 训练结束后对当前任务做准确率评估。
- 保存 LLM 和各 SLM 的可训练权重。

## 四、操作流程

以下操作都以项目根目录为基准。

### 1. 修改实验配置

主要配置文件：

```text
doc/tutorial/fedmkt/configs/default.yaml
```

常用修改项：

- `data.active_task`：选择任务，支持 `arc_c`、`arc_e`、`rte`、`boolq`、`cqa`。   对应于六个数据集
- `data.tasks.<task>.dataset_name`：任务对应的数据集名称。
- `data.tasks.<task>.data_dir`：预处理后数据保存目录，也是 `test.py` 读取的数据目录。
- `paths.llm_pretrained`：中心 LLM 模型路径。    在这里更换model，中心服务器三种异构的10B以上的model，每更换一个新的model，就要把对应的词表更换，详见default.yaml中的llm_to_slm_vocab_mappings和slm_to_llm_vocab_mappings
- `paths.slm_pretrained`：4 个 SLM 模型路径。
- `paths.vocab_mapping_dir`：词表映射文件目录。
- `training.global_epochs`：全局训练轮数。
- `training.batch_size`：梯度累积步数。
- `training.kd_alpha`：监督学习损失权重。
- `training.distill_loss_type`：蒸馏损失类型，当前支持 `ce`、`kl`。
- `training.distill_strategy`：teacher 分布选择策略，当前支持 `greater`、`weighted_mean`。
- `runtime.*_cuda_visible_devices`：不同 FATE 角色绑定的 GPU。
- `mmlcc.use_aggregation`：是否启用 MMLCC 风格的 SLM teacher 聚合。
- `wandb.mode` / `swanlab.mode`：日志记录模式。 默认swanlab

### 2. 准备 FedMKT 数据切分

数据准备脚本：

```text
doc/tutorial/fedmkt/prepare_fedmkt_data.py
```

默认按 `configs/default.yaml` 读取任务配置，将原始 HuggingFace 数据集切分成：

- `client_0`
- `client_1`
- `client_2`
- `client_3`
- `common`
- `validation`，如果原数据集存在
- `test`，如果原数据集存在

运行示例：

```powershell
cd doc\tutorial\fedmkt
python prepare_fedmkt_data.py --config configs\default.yaml --task arc_c
```

如果只想覆盖输出目录：

```powershell
cd doc\tutorial\fedmkt
python prepare_fedmkt_data.py --config configs\default.yaml --task arc_c --output_dir E:\data\fedmkt\arc_c
```

### 3. 检查词表映射文件

`test.py` 会从 `paths.vocab_mapping_dir` 读取 LLM 与 SLM 之间的词表映射文件。默认目录：

```text
doc/tutorial/fedmkt/mapping/
```

配置中默认期望包含：

```text
opt_to_llama.json
gpt2_to_llama.json
llama_small_to_llama.json
bloom_to_llama.json
llama_to_opt.json
llama_to_gpt2.json
llama_to_llama_small
llama_to_bloom.json
```

当前仓库中已存在的映射文件包括：

```text
llama_small_to_llama.json
llama_to_bloom.json
llama_to_gpt2.json
llama_to_opt.json
```

运行 `test.py` 前，需要确保配置里引用到的映射文件都真实存在，或者修改 `default.yaml` 中的 `paths.slm_to_llm_vocab_mappings` 和 `paths.llm_to_slm_vocab_mappings` 与本地文件保持一致。

### 4. 运行核心训练脚本

核心运行命令：

```powershell
cd doc\tutorial\fedmkt
python3 test.py   --parties arbiter:10002 guest:9998 host:9999 host:10000 host:10001   --log_level INFO
```

运行时 `test.py` 会打印：

- 项目源码路径 `PROJECT_PYTHON_DIR`
- 当前配置文件路径
- 当前任务和数据集 
- 数据目录
- 是否启用 MMLCC 聚合
- wandb/swanlab 模式
- arbiter 使用的 GPU
- LLM dtype 与 device map

### 5.数据记录
所有数据存放在swanlab/wandb中进行云同步，默认swanlab

### 6. 使用环境变量临时覆盖配置

`test.py` 支持用环境变量覆盖部分配置，适合临时切换任务或调试小规模样本。

示例：只跑少量验证样本：

```powershell
$env:FEDMKT_EVAL_MAX_EXAMPLES="50"
cd doc\tutorial\fedmkt
python test.py
```

示例：临时切换数据目录：

```powershell
$env:FEDMKT_DATA_DIR="E:\data\fedmkt\arc_c"
cd doc\tutorial\fedmkt
python test.py
```

常用环境变量：

```text
FEDMKT_CONFIG
FEDMKT_TASK
FEDMKT_DATASET_NAME
FEDMKT_DATA_DIR
FEDMKT_EVAL_SPLIT
FEDMKT_EVAL_MAX_EXAMPLES
FEDMKT_PUBLIC_SELECT_NUM
FEDMKT_PUBLIC_START_IDX
WANDB_MODE
WANDB_PROJECT
WANDB_GROUP
SWANLAB_MODE
SWANLAB_PROJECT
SWANLAB_GROUP
SWANLAB_WORKSPACE
SWANLAB_LOGDIR
```

## 五、训练产物

默认保存路径由 `configs/default.yaml` 控制：

```text
paths.llm_model_save_dir
paths.slm_model_save_dirs
paths.training_output_dir
```

默认值：

```text
doc/tutorial/fedmkt/models/fedmkt_4_slms_llm_model
doc/tutorial/fedmkt/models/fedmkt_4_slms_slm_0
doc/tutorial/fedmkt/models/fedmkt_4_slms_slm_1
doc/tutorial/fedmkt/models/fedmkt_4_slms_slm_2
doc/tutorial/fedmkt/models/fedmkt_4_slms_slm_3
```

因为 `test.py` 中 `save_trainable_weights_only=True`，保存内容主要是 LoRA/可训练权重，而不是完整基础模型。

## 六、关键代码链路

```text
doc/tutorial/fedmkt/test.py
  ├── train_llm(ctx)
  │   ├── fate_llm.model_zoo.pellm.llama.LLaMa
  │   ├── fate_llm.dataset.qa_dataset.QaDataset
  │   ├── fate_llm.algo.fedmkt.FedMKTTrainingArguments
  │   └── fate_llm.algo.fedmkt.FedMKTLLM
  ├── train_slm(ctx, slm_idx)
  │   ├── fate_llm.model_zoo.pellm.opt.OPT
  │   ├── fate_llm.model_zoo.pellm.gpt2.GPT2CLM
  │   ├── fate_llm.model_zoo.pellm.llama.LLaMa
  │   ├── fate_llm.model_zoo.pellm.bloom.Bloom
  │   ├── fate_llm.dataset.qa_dataset.QaDataset
  │   └── fate_llm.algo.fedmkt.FedMKTSLM
  └── evaluate_task_accuracy(...)
```

FedMKT 训练主体：

```text
python/fate_llm/algo/fedmkt/fedmkt.py
python/fate_llm/algo/fedmkt/fedmkt_trainer.py
python/fate_llm/algo/fedmkt/token_alignment/
python/fate_llm/algo/fedmkt/utils/
python/fate_llm/algo/fedmkt/mmlcc.py
```

## 七、注意事项

- `test.py` 依赖从 `doc/tutorial/fedmkt` 目录运行时的相对路径，建议进入该目录后执行。
- `configs/default.yaml` 中模型路径和数据路径带有本地绝对路径，需要按实际机器调整。
- `mapping` 目录里的词表映射文件必须和配置中的文件名一致。
- `models/` 目录当前只看到 `llama-2-7b-hf` 和 `opt-1.3b`，而默认配置还引用了 `gpt2-xl`、`Sheared-LLaMA-1.3B`、`bloom-1b1` 等路径，运行前需要保证这些路径可用或修改配置。
- 根目录 `main.py` 是一个 tokenizer 对比小脚本，不是 FedMKT 主入口。

## 去噪实验流程

主要配置文件：

```text
doc/tutorial/fedmkt/configs/default.yaml
```

修改default.yaml后，先准备干净数据集：

```powershell
cd doc/tutorial/fedmkt
python prepare_fedmkt_data.py --config configs/default.yaml --task arc_c
```

然后生成噪声数据集：

```powershell
python add_arc_noise.py --config configs/default.yaml
```

如需下载模型：
```powershell
python download_models.py
```

最后运行测试脚本（baseline和去噪版本）：
```powershell
python3 test.py \
  --parties arbiter:10002 guest:9998 host:9999 host:10000 host:10001 \
  --log_level INFO
python3 test_gmm_denoise.py \
  --parties arbiter:10002 guest:9998 host:9999 host:10000 host:10001 \
  --log_level INFO
```