# fedmkt_4_slms.py

import os
import json
import math
import ast
import sys
import warnings

from fate.arch.launchers.multiprocess_launcher import launch
from vocab_mapping_selector import select_vocab_mapping_names

wandb = None
swanlab = None


warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Could not find a config file in .* - will assume that the vocabulary was not modified.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Parameter 'fn_kwargs'.*couldn't be hashed properly.*",
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PROJECT_PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PROJECT_PYTHON_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PYTHON_DIR)

from fate_llm.algo.fedmkt.utils.local_metric_logger import (
    configure_local_tracking,
    local_tracking_enabled,
    log_local_metrics,
    set_local_run_name,
)

torch = None


def ensure_torch_imported():
    global torch
    if torch is None:
        import torch as torch_module
        torch = torch_module
    return torch


def ensure_wandb_imported():
    global wandb
    if wandb is None:
        try:
            import wandb as wandb_module
            wandb = wandb_module
        except ImportError:
            wandb = False
    return None if wandb is False else wandb


def ensure_swanlab_imported():
    global swanlab
    if swanlab is None:
        try:
            import swanlab as swanlab_module
            swanlab = swanlab_module
        except ImportError:
            swanlab = False
    return None if swanlab is False else swanlab


def _env_optional_int(name):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _parse_bool(value, default=False):
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if normalized in {"", "0", "false", "no", "n", "off", "disabled", "none"}:
        return False
    return default


def _normalize_dataset_name(name):
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "arc_e": "arc_easy",
        "arc_easy": "arc_easy",
        "arc_challenge": "arc_challenge",
        "arc_c": "arc_challenge",
        "boolq": "boolq",
        "bool_q": "boolq",
        "cqa": "commonsenseqa",
        "commonsenseqa": "commonsenseqa",
        "commonsense_qa": "commonsenseqa",
        "rte": "rte",
    }
    return aliases.get(normalized, normalized)


process_data_output_dir = os.environ.get("FEDMKT_DATA_DIR", "../../../../.")
fedmkt_dataset_name = _normalize_dataset_name(os.environ.get("FEDMKT_DATASET_NAME", "arc_challenge"))
eval_split = os.environ.get("FEDMKT_EVAL_SPLIT", "validation")
eval_max_examples = _env_optional_int("FEDMKT_EVAL_MAX_EXAMPLES")
public_select_num = _env_optional_int("FEDMKT_PUBLIC_SELECT_NUM")
public_start_idx = _env_optional_int("FEDMKT_PUBLIC_START_IDX")
# 修改成绝对路径
llm_pretrained_path = "../../../models/llama-2-7b-hf"
slm_0_pretrained_path = "../../../models/opt-1.3b"
slm_1_pretrained_path = "../../../models/gpt2-xl"
slm_2_pretrained_path = "../../../models/Sheared-LLaMA-1.3B"
slm_3_pretrained_path = "../../../models/bloom-1b1"

vocab_mapping_directory = "./mapping"

"""
slm_to_llm_vocab_mapping_paths = [
    "opt_to_llama.json",
    "gpt2_to_llama.json",
    "llama_small_to_llama.json",
    "bloom_to_llama.json",
]
llm_to_slm_vocab_mapping_paths = [
    "llama_to_opt.json",
    "llama_to_gpt2.json",
    "llama_to_llama_small",
    "llama_to_bloom.json",
]

for idx in range(4):
    slm_to_llm_vocab_mapping_paths[idx] = vocab_mapping_directory + "/" + slm_to_llm_vocab_mapping_paths[idx]
    llm_to_slm_vocab_mapping_paths[idx] = vocab_mapping_directory + "/" + llm_to_slm_vocab_mapping_paths[idx]
"""

slm_pretrained_paths = [
    slm_0_pretrained_path,
    slm_1_pretrained_path,
    slm_2_pretrained_path,
    slm_3_pretrained_path,
]

slm_lora_target_modules = [
    ["q_proj", "v_proj"],
    ["c_attn"],
    ["q_proj", "k_proj", "v_proj", "o_proj"],
    ["query_key_value"],
]

global_epochs = 100
batch_size = 8
llm_lr = 3e-5
slm_lrs = [3e-5, 3e-4, 3e-5, 3e-5, 3e-5]

llm_model_saved_directory = "./models/fedmkt_4_slms_llm_model"
slm_models_saved_directory = [
    "./models/fedmkt_4_slms_slm_0",
    "./models/fedmkt_4_slms_slm_1",
    "./models/fedmkt_4_slms_slm_2",
    "./models/fedmkt_4_slms_slm_3",
]


def _load_experiment_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fin:
        if path.endswith(".json"):
            return json.load(fin) or {}
        text = fin.read()
        try:
            import yaml
        except ImportError as exc:
            return _load_simple_yaml(text)
        return yaml.safe_load(text) or {}


def _parse_simple_yaml_scalar(value):
    value = value.strip()
    if value == "" or value.lower() in {"null", "none", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip('"').strip("'")


def _strip_yaml_comment(line):
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _load_simple_yaml(text):
    cleaned = []
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        cleaned.append((len(line) - len(line.lstrip(" ")), line.strip()))

    root = {}
    stack = [(-1, root)]
    for idx, (indent, content) in enumerate(cleaned):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"unsupported YAML list placement near: {content}")
            parent.append(_parse_simple_yaml_scalar(content[2:]))
            continue

        if ":" not in content:
            raise ValueError(f"unsupported YAML line: {content}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            parent[key] = _parse_simple_yaml_scalar(value)
            continue

        next_is_list = False
        if idx + 1 < len(cleaned):
            next_indent, next_content = cleaned[idx + 1]
            next_is_list = next_indent > indent and next_content.startswith("- ")
        container = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))

    return root


def _cfg(config, path, default=None):
    cur = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _cfg_env(config, name, path, default=None, cast=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        value = _cfg(config, path, default)
    if cast is not None and value is not None:
        return cast(value)
    return value


def _normalize_task_name(name):
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "arc_c": "arc_c",
        "arc_challenge": "arc_c",
        "arc_e": "arc_e",
        "arc_easy": "arc_e",
        "rte": "rte",
        "boolq": "boolq",
        "bool_q": "boolq",
        "cqa": "cqa",
        "commonsenseqa": "cqa",
        "commonsense_qa": "cqa",
    }
    return aliases.get(normalized, normalized)


def _resolve_data_settings(config, default_dataset_name, default_data_dir):
    active_task = _normalize_task_name(os.environ.get("FEDMKT_TASK", _cfg(config, "data.active_task", "arc_c")))
    task_config = _cfg(config, f"data.tasks.{active_task}", {})

    dataset_name = os.environ.get("FEDMKT_DATASET_NAME")
    if dataset_name is None or dataset_name.strip() == "":
        dataset_name = task_config.get("dataset_name") if isinstance(task_config, dict) else None
    if dataset_name is None or str(dataset_name).strip() == "":
        dataset_name = _cfg(config, "data.dataset_name", default_dataset_name)

    data_dir = os.environ.get("FEDMKT_DATA_DIR")
    if data_dir is None or data_dir.strip() == "":
        data_dir = task_config.get("data_dir") if isinstance(task_config, dict) else None
    if data_dir is None or str(data_dir).strip() == "":
        data_dir = _cfg(config, "data.data_dir", default_data_dir)

    return active_task, _normalize_dataset_name(dataset_name), data_dir


def _join_mapping_paths(directory, filenames):
    return [os.path.join(directory, filename) for filename in filenames]


experiment_config_path = os.environ.get(
    "FEDMKT_CONFIG",
    os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
)
experiment_config = _load_experiment_config(experiment_config_path)

fedmkt_active_task, fedmkt_dataset_name, process_data_output_dir = _resolve_data_settings(
    experiment_config,
    default_dataset_name=fedmkt_dataset_name,
    default_data_dir=process_data_output_dir,
)
eval_split = _cfg_env(experiment_config, "FEDMKT_EVAL_SPLIT", "data.eval_split", eval_split)
eval_max_examples = _cfg_env(experiment_config, "FEDMKT_EVAL_MAX_EXAMPLES", "data.eval_max_examples", eval_max_examples, int)
public_select_num = _cfg_env(experiment_config, "FEDMKT_PUBLIC_SELECT_NUM", "data.public_select_num", public_select_num, int)
public_start_idx = _cfg_env(experiment_config, "FEDMKT_PUBLIC_START_IDX", "data.public_start_idx", public_start_idx, int)

llm_pretrained_path = _cfg(experiment_config, "paths.llm_pretrained", llm_pretrained_path)
slm_pretrained_paths = _cfg(experiment_config, "paths.slm_pretrained", slm_pretrained_paths)
vocab_mapping_directory = _cfg(experiment_config, "paths.vocab_mapping_dir", vocab_mapping_directory)
slm_to_llm_mapping_names, llm_to_slm_mapping_names = select_vocab_mapping_names(llm_pretrained_path)
slm_to_llm_vocab_mapping_paths = _join_mapping_paths(
    vocab_mapping_directory,
    slm_to_llm_mapping_names,
)
llm_to_slm_vocab_mapping_paths = _join_mapping_paths(
    vocab_mapping_directory,
    llm_to_slm_mapping_names,
)
training_output_dir = _cfg(experiment_config, "paths.training_output_dir", "../../../../.")
llm_model_saved_directory = _cfg(experiment_config, "paths.llm_model_save_dir", llm_model_saved_directory)
slm_models_saved_directory = _cfg(experiment_config, "paths.slm_model_save_dirs", slm_models_saved_directory)


def _validate_model_path(model_path, label):
    if not isinstance(model_path, str) or not os.path.isabs(model_path):
        return
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"{label} local model path does not exist: {model_path}. "
            "If this is a Hugging Face repo, use the form 'namespace/repo_name' instead of an absolute path."
        )
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"{label} model path is missing config.json: {model_path}")


_validate_model_path(llm_pretrained_path, "LLM")
for _slm_idx, _slm_path in enumerate(slm_pretrained_paths):
    _validate_model_path(_slm_path, f"SLM[{_slm_idx}]")

slm_lora_target_modules = _cfg(experiment_config, "lora.slm_target_modules", slm_lora_target_modules)
llm_lora_target_modules = _cfg(experiment_config, "lora.llm_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
lora_r = int(_cfg(experiment_config, "lora.r", 8))
lora_alpha = int(_cfg(experiment_config, "lora.alpha", 16))
lora_dropout = float(_cfg(experiment_config, "lora.dropout", 0.05))

global_epochs = int(_cfg(experiment_config, "training.global_epochs", global_epochs))
batch_size = int(_cfg(experiment_config, "training.batch_size", batch_size))
llm_lr = float(_cfg(experiment_config, "training.llm_lr", llm_lr))
slm_lrs = _cfg(experiment_config, "training.slm_lrs", slm_lrs)
kd_alpha = float(_cfg(experiment_config, "training.kd_alpha", 0.9))
distill_loss_type = _cfg(experiment_config, "training.distill_loss_type", "ce")
distill_strategy = _cfg(experiment_config, "training.distill_strategy", "greater")
distill_temperature = float(_cfg(experiment_config, "training.distill_temperature", 1.0))
dataloader_num_workers = int(_cfg(experiment_config, "training.dataloader_num_workers", 4))
warmup_ratio = float(_cfg(experiment_config, "training.warmup_ratio", 0.008))
lr_scheduler_type = _cfg(experiment_config, "training.lr_scheduler_type", "cosine")
optim = _cfg(experiment_config, "training.optim", "adamw_torch")
adam_beta1 = float(_cfg(experiment_config, "training.adam_beta1", 0.9))
adam_beta2 = float(_cfg(experiment_config, "training.adam_beta2", 0.95))
weight_decay = float(_cfg(experiment_config, "training.weight_decay", 0.1))
max_grad_norm = float(_cfg(experiment_config, "training.max_grad_norm", 1.0))
per_device_train_batch_size = int(_cfg(experiment_config, "training.per_device_train_batch_size", 1))
use_cpu = bool(_cfg(experiment_config, "training.use_cpu", False))
llm_torch_dtype = _cfg(experiment_config, "training.llm_torch_dtype", "float32")
slm_torch_dtype = _cfg(experiment_config, "training.slm_torch_dtype", "float32")
llm_device_map = _cfg(experiment_config, "training.llm_device_map", None)
llm_low_cpu_mem_usage = bool(_cfg(experiment_config, "training.llm_low_cpu_mem_usage", False))
llm_max_memory = _cfg(experiment_config, "training.llm_max_memory", None)
llm_model_load_kwargs = {}
if llm_device_map not in {None, "", "none", "None"}:
    llm_model_load_kwargs["device_map"] = llm_device_map
if llm_low_cpu_mem_usage:
    llm_model_load_kwargs["low_cpu_mem_usage"] = True
if llm_max_memory:
    llm_model_load_kwargs["max_memory"] = {int(k): str(v) for k, v in llm_max_memory.items()}

arbiter_cuda_visible_devices = str(_cfg(experiment_config, "runtime.arbiter_cuda_visible_devices", "2"))
guest_cuda_visible_devices = str(_cfg(experiment_config, "runtime.guest_cuda_visible_devices", "4"))
host_cuda_visible_devices = _cfg(
    experiment_config,
    "runtime.host_cuda_visible_devices",
    {"9999": "5", "10000": "6", "10001": "7"},
)

use_mmlcc_aggregation = bool(_cfg(experiment_config, "mmlcc.use_aggregation", False))
mmlcc_probability_epsilon = float(_cfg(experiment_config, "mmlcc.probability_epsilon", 1e-12))
mmlcc_num_blocks = int(_cfg(experiment_config, "mmlcc.num_blocks", 1))
mmlcc_privacy_guarantee = int(_cfg(experiment_config, "mmlcc.privacy_guarantee", 1))
mmlcc_beta_radius = float(_cfg(experiment_config, "mmlcc.beta_radius", 1.15))
mmlcc_noise_sigma = float(_cfg(experiment_config, "mmlcc.noise_sigma", 1.0))
mmlcc_noise_clip_theta = float(_cfg(experiment_config, "mmlcc.noise_clip_theta", 6.0))
mmlcc_seed = int(_cfg(experiment_config, "mmlcc.seed", 42))


# GMM denoising config.  By default this test uses the noisy dataset generated by
# add_arc_noise.py while keeping the same FedMKT split names.
denoise_use_noisy_data = _parse_bool(
    os.environ.get("FEDMKT_USE_NOISY_DATA"),
    _parse_bool(_cfg(experiment_config, "denoise.use_noisy_data", True), True),
)
denoise_noisy_data_dir = os.environ.get("FEDMKT_NOISY_DATA_DIR") or _cfg(experiment_config, "denoise.noisy_data_dir", None) or _cfg(experiment_config, "noise.output_dir", None)
if denoise_use_noisy_data and denoise_noisy_data_dir:
    process_data_output_dir = denoise_noisy_data_dir

gmm_enabled = _parse_bool(os.environ.get("FEDMKT_GMM_ENABLED"), _parse_bool(_cfg(experiment_config, "denoise.gmm_enabled", True), True))
gmm_warmup_steps = int(_cfg_env(experiment_config, "FEDMKT_GMM_WARMUP_STEPS", "denoise.gmm_warmup_steps", 20, int))
gmm_update_interval = int(_cfg_env(experiment_config, "FEDMKT_GMM_UPDATE_INTERVAL", "denoise.gmm_update_interval", 5, int))
gmm_history_size = int(_cfg_env(experiment_config, "FEDMKT_GMM_HISTORY_SIZE", "denoise.gmm_history_size", 4096, int))
gmm_min_samples = int(_cfg_env(experiment_config, "FEDMKT_GMM_MIN_SAMPLES", "denoise.gmm_min_samples", 64, int))
gmm_clean_threshold = float(_cfg_env(experiment_config, "FEDMKT_GMM_CLEAN_THRESHOLD", "denoise.gmm_clean_threshold", 0.5, float))
gmm_noisy_weight = float(_cfg_env(experiment_config, "FEDMKT_GMM_NOISY_WEIGHT", "denoise.gmm_noisy_weight", 0.2, float))
gmm_weight_power = float(_cfg_env(experiment_config, "FEDMKT_GMM_WEIGHT_POWER", "denoise.gmm_weight_power", 1.0, float))
denoise_loss_type = str(_cfg_env(experiment_config, "FEDMKT_DENOISE_LOSS_TYPE", "denoise.loss_type", "ce"))
sce_alpha = float(_cfg_env(experiment_config, "FEDMKT_SCE_ALPHA", "denoise.sce_alpha", 1.0, float))
sce_beta = float(_cfg_env(experiment_config, "FEDMKT_SCE_BETA", "denoise.sce_beta", 0.1, float))
rce_epsilon = float(_cfg_env(experiment_config, "FEDMKT_RCE_EPSILON", "denoise.rce_epsilon", 1.0e-4, float))
denoise_loss_log_every_n_steps = int(_cfg_env(experiment_config, "FEDMKT_DENOISE_LOSS_LOG_EVERY_N_STEPS", "denoise.loss_log_every_n_steps", 20, int))

wandb_mode = str(_cfg_env(experiment_config, "WANDB_MODE", "wandb.mode", "disabled")).lower()
wandb_project = str(_cfg_env(experiment_config, "WANDB_PROJECT", "wandb.project", "fedmkt"))
wandb_group = str(_cfg_env(experiment_config, "WANDB_GROUP", "wandb.group", "fedmkt"))
wandb_init_timeout = int(_cfg_env(experiment_config, "WANDB_INIT_TIMEOUT", "wandb.init_timeout", 30, int))
wandb_enabled = wandb_mode not in {"", "disabled", "disable", "false", "0", "none"}
if not wandb_enabled:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
else:
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = wandb_mode

swanlab_mode = str(_cfg_env(experiment_config, "SWANLAB_MODE", "swanlab.mode", "disabled")).lower()
swanlab_project = str(_cfg_env(experiment_config, "SWANLAB_PROJECT", "swanlab.project", "fedmkt"))
swanlab_group = _cfg_env(experiment_config, "SWANLAB_GROUP", "swanlab.group", "fedmkt")
swanlab_workspace = _cfg_env(experiment_config, "SWANLAB_WORKSPACE", "swanlab.workspace", None)
swanlab_logdir = _cfg_env(experiment_config, "SWANLAB_LOGDIR", "swanlab.logdir", "swanlog")
swanlab_enabled = swanlab_mode not in {"", "disabled", "disable", "false", "0", "none"}
os.environ["SWANLAB_MODE"] = swanlab_mode

local_tracking_enabled_cfg = _parse_bool(
    os.environ.get("FEDMKT_LOCAL_TRACKING_ENABLED"),
    _parse_bool(_cfg(experiment_config, "local_tracking.enabled", True), True),
)
local_tracking_log_dir = _cfg_env(
    experiment_config,
    "FEDMKT_LOCAL_LOG_DIR",
    "local_tracking.log_dir",
    os.path.join(PROJECT_ROOT, "local_metrics"),
)
if local_tracking_log_dir and not os.path.isabs(str(local_tracking_log_dir)):
    local_tracking_log_dir = os.path.abspath(os.path.join(PROJECT_ROOT, str(local_tracking_log_dir)))
local_tracking_write_jsonl = _parse_bool(
    os.environ.get("FEDMKT_LOCAL_WRITE_JSONL"),
    _parse_bool(_cfg(experiment_config, "local_tracking.write_jsonl", True), True),
)
local_tracking_write_csv = _parse_bool(
    os.environ.get("FEDMKT_LOCAL_WRITE_CSV"),
    _parse_bool(_cfg(experiment_config, "local_tracking.write_csv", True), True),
)
configure_local_tracking(
    enabled=local_tracking_enabled_cfg,
    log_dir=local_tracking_log_dir,
    run_name="fedmkt",
    write_jsonl=local_tracking_write_jsonl,
    write_csv=local_tracking_write_csv,
)


def _wandb_active():
    wandb_module = ensure_wandb_imported() if wandb_enabled else None
    return wandb_module is not None and getattr(wandb_module, "run", None) is not None


def print_cuda_memory(prefix):
    torch = ensure_torch_imported()
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(idx) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(idx) / 1024 ** 3
        free, total = torch.cuda.mem_get_info(idx)
        print(
            f"[cuda][{prefix}] cuda:{idx} "
            f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
            f"free={free / 1024 ** 3:.2f}GiB total={total / 1024 ** 3:.2f}GiB",
            flush=True,
        )


def print_cuda_binding(ctx, prefix):
    torch = ensure_torch_imported()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    initialized_before = torch.cuda.is_initialized()
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(idx) for idx in range(device_count)]
    else:
        device_count = 0
        device_names = []
    print(
        f"[cuda-binding][{prefix}] party={ctx.local.party} "
        f"CUDA_VISIBLE_DEVICES={cuda_visible} "
        f"cuda_initialized_before_check={initialized_before} "
        f"torch_device_count={device_count} "
        f"device_names={device_names}",
        flush=True,
    )
    return initialized_before, device_count


def print_model_device(prefix, model):
    try:
        first_param_device = next(model.parameters()).device
    except StopIteration:
        first_param_device = "no_parameters"
    except Exception as exc:
        first_param_device = f"unavailable:{exc}"
    print(
        f"[model-device][{prefix}] first_parameter_device={first_param_device} "
        f"hf_device_map={getattr(model, 'hf_device_map', None)}",
        flush=True,
    )


def _parse_cuda_visible_devices(value):
    devices = []
    for raw in str(value or "").split(","):
        raw = raw.strip()
        if raw == "":
            continue
        try:
            devices.append(int(raw))
        except ValueError:
            pass
    return devices


def configure_process_cuda(ctx, prefix, visible_devices, force_single_device=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    initialized_before, device_count = print_cuda_binding(ctx, prefix)
    selected_devices = _parse_cuda_visible_devices(visible_devices)
    os.environ.pop("FEDMKT_FORCE_CUDA_DEVICE", None)

    if initialized_before and force_single_device and selected_devices:
        torch = ensure_torch_imported()
        target_device = selected_devices[0]
        if target_device < device_count:
            torch.cuda.set_device(target_device)
            os.environ["FEDMKT_FORCE_CUDA_DEVICE"] = str(target_device)
            print(
                f"[cuda-binding][{prefix}] CUDA was initialized before binding; "
                f"forcing this process to physical cuda:{target_device}",
                flush=True,
            )
        else:
            print(
                f"[cuda-binding][{prefix}] cannot force cuda:{target_device}; "
                f"torch_device_count={device_count}",
                flush=True,
            )

    if initialized_before and not force_single_device and selected_devices:
        configure_llm_max_memory_for_preinitialized_cuda(selected_devices, device_count, prefix)


def configure_llm_max_memory_for_preinitialized_cuda(selected_devices, device_count, prefix):
    if llm_model_load_kwargs.get("device_map") in {None, "", "none", "None"}:
        return
    if llm_model_load_kwargs.get("max_memory"):
        return
    if not selected_devices:
        return
    max_memory = {}
    selected_set = set(selected_devices)
    for idx in range(device_count):
        max_memory[idx] = "38GiB" if idx in selected_set else "0GiB"
    llm_model_load_kwargs["max_memory"] = max_memory
    print(
        f"[cuda-binding][{prefix}] CUDA was initialized before binding; "
        f"restricting LLM device_map with max_memory={max_memory}",
        flush=True,
    )


def force_training_args_device(training_args):
    forced_device = os.environ.get("FEDMKT_FORCE_CUDA_DEVICE")
    if forced_device in {None, ""}:
        return training_args
    torch = ensure_torch_imported()
    device = torch.device(f"cuda:{int(forced_device)}")
    torch.cuda.set_device(device)
    training_args.__dict__["_setup_devices"] = device
    training_args._n_gpu = 1
    print(
        f"[training-args-device] forced training_args.device={device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    return training_args


def _swanlab_active():
    return ensure_swanlab_imported() is not None if swanlab_enabled else False


def _tracking_active():
    return _wandb_active() or _swanlab_active() or local_tracking_enabled()


def _tracking_define_metric(*args, **kwargs):
    if _wandb_active():
        try:
            ensure_wandb_imported().define_metric(*args, **kwargs)
        except Exception:
            pass


def _tracking_log(metrics):
    log_local_metrics(metrics)
    if _wandb_active():
        try:
            ensure_wandb_imported().log(metrics)
        except Exception as exc:
            print(f"[wandb] log skipped after failure: {exc}", flush=True)
    if _swanlab_active():
        try:
            ensure_swanlab_imported().log(metrics)
        except Exception as exc:
            print(f"[swanlab] log skipped after failure: {exc}", flush=True)


def ensure_wandb_run(ctx, name: str):
    """
    多进程下不要依赖 report_to 自动 init，强制确保当前进程里 wandb.run 可用
    """
    global wandb_enabled, swanlab_enabled
    set_local_run_name(name)
    swanlab_module = ensure_swanlab_imported() if swanlab_enabled else None
    if swanlab_enabled and swanlab_module is not None:
        try:
            swanlab_module.init(
                project=swanlab_project,
                workspace=swanlab_workspace,
                experiment_name=name,
                group=swanlab_group,
                mode=swanlab_mode,
                logdir=swanlab_logdir,
                config=experiment_config,
                reinit=True,
            )
        except Exception as exc:
            swanlab_enabled = False
            print(f"[swanlab] disabled after init failure: {exc}", flush=True)

    wandb_module = ensure_wandb_imported() if wandb_enabled else None
    if not wandb_enabled or wandb_module is None or getattr(wandb_module, "run", None) is not None:
        return

    try:
        wandb_module.init(
            project=wandb_project,
            group=wandb_group,
            name=name,
            reinit=True,
            mode=wandb_mode,
            settings=wandb_module.Settings(init_timeout=wandb_init_timeout),
        )
    except Exception as exc:
        wandb_enabled = False
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        print(f"[wandb] disabled after init failure: {exc}", flush=True)
        return
    for prefix in ("client", "server"):
        _tracking_define_metric(f"{prefix}/fedmkt_round")
        _tracking_define_metric(f"{prefix}/fedmkt_loss_step")
        _tracking_define_metric(f"{prefix}/total_loss", step_metric=f"{prefix}/fedmkt_loss_step")
        _tracking_define_metric(f"{prefix}/supervised_lm_loss", step_metric=f"{prefix}/fedmkt_loss_step")
        _tracking_define_metric(f"{prefix}/distill_loss", step_metric=f"{prefix}/fedmkt_loss_step")
        _tracking_define_metric(f"{prefix}/weighted_supervised_lm_loss", step_metric=f"{prefix}/fedmkt_loss_step")
        _tracking_define_metric(f"{prefix}/weighted_distill_loss", step_metric=f"{prefix}/fedmkt_loss_step")
        _tracking_define_metric(f"{prefix}/arc_mc_accuracy", step_metric=f"{prefix}/fedmkt_round")
        _tracking_define_metric(f"{prefix}/arc_mc_avg_choice_loss", step_metric=f"{prefix}/fedmkt_round")
        _tracking_define_metric(f"{prefix}/arc_mc_num_examples", step_metric=f"{prefix}/fedmkt_round")


def evaluate_arc_mc_accuracy(model, tokenizer, split="validation", max_examples=None, arc_config="ARC-Challenge"):
    import torch
    from datasets import load_dataset

    ds = load_dataset("ai2_arc", arc_config, split=split)

    model.eval()
    device = next(model.parameters()).device

    num_examples = 0
    num_correct = 0
    nll_sum = 0.0

    for i, ex in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break

        question = ex["question"]
        choices_text = ex["choices"]["text"]
        choices_label = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        try:
            gold_idx = choices_label.index(answer_key)
        except ValueError:
            continue

        lines = [f"Question: {question}"]
        for lab, txt in zip(choices_label, choices_text):
            lines.append(f"{lab}. {txt}")
        lines.append("Answer:")
        prompt = "\n".join(lines)

        with torch.no_grad():
            choice_nlls = []
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = int(prompt_enc["input_ids"].shape[1])

            for choice in choices_text:
                full_text = prompt + " " + choice
                full_enc = tokenizer(full_text, return_tensors="pt")

                input_ids = full_enc["input_ids"].to(device)
                attention_mask = full_enc["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                choice_nlls.append(float(outputs.loss.item()))

        num_examples += 1
        nll_sum += choice_nlls[gold_idx]

        pred_idx = int(min(range(len(choice_nlls)), key=lambda k: choice_nlls[k]))
        if pred_idx == gold_idx:
            num_correct += 1

    if num_examples == 0:
        return {
            "arc_mc_accuracy": 0.0,
            "arc_mc_avg_neg_loglik": float("inf"),
            "arc_mc_num_examples": 0.0,
        }

    return {
        "arc_mc_accuracy": float(num_correct) / float(num_examples),
        "arc_mc_avg_neg_loglik": float(nll_sum) / float(num_examples),
        "arc_mc_num_examples": float(num_examples),
    }


def evaluate_rte_accuracy(model, tokenizer, split="validation", max_examples=None):
    import torch
    from datasets import load_dataset

    ds = load_dataset("glue", "rte", split=split)

    model.eval()
    device = next(model.parameters()).device

    num_examples = 0
    num_correct = 0
    nll_sum = 0.0
    choices = [" True", " False"]

    for i, ex in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        if int(ex["label"]) < 0:
            continue

        prompt = f"{ex['sentence1']}\nQuestion: {ex['sentence2']} True or False?\nAnswer:"
        gold_idx = int(ex["label"])

        with torch.no_grad():
            choice_nlls = []
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = int(prompt_enc["input_ids"].shape[1])

            for choice in choices:
                full_text = prompt + choice
                full_enc = tokenizer(full_text, return_tensors="pt")

                input_ids = full_enc["input_ids"].to(device)
                attention_mask = full_enc["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                choice_nlls.append(float(outputs.loss.item()))

        num_examples += 1
        nll_sum += choice_nlls[gold_idx]

        pred_idx = int(min(range(len(choice_nlls)), key=lambda k: choice_nlls[k]))
        if pred_idx == gold_idx:
            num_correct += 1

    if num_examples == 0:
        return {
            "rte_accuracy": 0.0,
            "rte_avg_neg_loglik": float("inf"),
            "rte_num_examples": 0.0,
        }

    return {
        "rte_accuracy": float(num_correct) / float(num_examples),
        "rte_avg_neg_loglik": float(nll_sum) / float(num_examples),
        "rte_num_examples": float(num_examples),
    }


def evaluate_boolq_accuracy(model, tokenizer, split="validation", max_examples=None):
    import torch
    from datasets import load_dataset

    ds = load_dataset("super_glue", "boolq", split=split)

    model.eval()
    device = next(model.parameters()).device

    num_examples = 0
    num_correct = 0
    nll_sum = 0.0
    choices = [" yes", " no"]

    for i, ex in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break

        prompt = f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:"
        if "answer" in ex:
            gold_idx = 0 if bool(ex["answer"]) else 1
        else:
            if int(ex["label"]) < 0:
                continue
            gold_idx = 0 if int(ex["label"]) == 1 else 1

        with torch.no_grad():
            choice_nlls = []
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = int(prompt_enc["input_ids"].shape[1])

            for choice in choices:
                full_text = prompt + choice
                full_enc = tokenizer(full_text, return_tensors="pt")

                input_ids = full_enc["input_ids"].to(device)
                attention_mask = full_enc["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                choice_nlls.append(float(outputs.loss.item()))

        num_examples += 1
        nll_sum += choice_nlls[gold_idx]

        pred_idx = int(min(range(len(choice_nlls)), key=lambda k: choice_nlls[k]))
        if pred_idx == gold_idx:
            num_correct += 1

    if num_examples == 0:
        return {
            "boolq_accuracy": 0.0,
            "boolq_avg_neg_loglik": float("inf"),
            "boolq_num_examples": 0.0,
        }

    return {
        "boolq_accuracy": float(num_correct) / float(num_examples),
        "boolq_avg_neg_loglik": float(nll_sum) / float(num_examples),
        "boolq_num_examples": float(num_examples),
    }


def evaluate_commonsenseqa_accuracy(model, tokenizer, split="validation", max_examples=None):
    import torch
    from datasets import load_dataset

    ds = load_dataset("commonsense_qa", split=split)

    model.eval()
    device = next(model.parameters()).device

    num_examples = 0
    num_correct = 0
    nll_sum = 0.0

    for i, ex in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break

        choices_text = ex["choices"]["text"]
        choices_label = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        try:
            gold_idx = choices_label.index(answer_key)
        except ValueError:
            continue

        prompt = f"Question: {ex['question']}\nAnswer:"

        with torch.no_grad():
            choice_nlls = []
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = int(prompt_enc["input_ids"].shape[1])

            for choice in choices_text:
                full_text = prompt + " " + choice
                full_enc = tokenizer(full_text, return_tensors="pt")

                input_ids = full_enc["input_ids"].to(device)
                attention_mask = full_enc["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                choice_nlls.append(float(outputs.loss.item()))

        num_examples += 1
        nll_sum += choice_nlls[gold_idx]

        pred_idx = int(min(range(len(choice_nlls)), key=lambda k: choice_nlls[k]))
        if pred_idx == gold_idx:
            num_correct += 1

    if num_examples == 0:
        return {
            "commonsenseqa_accuracy": 0.0,
            "commonsenseqa_avg_neg_loglik": float("inf"),
            "commonsenseqa_num_examples": 0.0,
        }

    return {
        "commonsenseqa_accuracy": float(num_correct) / float(num_examples),
        "commonsenseqa_avg_neg_loglik": float(nll_sum) / float(num_examples),
        "commonsenseqa_num_examples": float(num_examples),
    }


def evaluate_task_accuracy(model, tokenizer, dataset_name, split="validation", max_examples=None):
    if dataset_name == "arc_challenge":
        return evaluate_arc_mc_accuracy(
            model,
            tokenizer,
            split=split,
            max_examples=max_examples,
            arc_config="ARC-Challenge",
        )
    if dataset_name == "arc_easy":
        return evaluate_arc_mc_accuracy(
            model,
            tokenizer,
            split=split,
            max_examples=max_examples,
            arc_config="ARC-Easy",
        )
    if dataset_name == "rte":
        return evaluate_rte_accuracy(model, tokenizer, split=split, max_examples=max_examples)
    if dataset_name == "boolq":
        return evaluate_boolq_accuracy(model, tokenizer, split=split, max_examples=max_examples)
    if dataset_name == "commonsenseqa":
        return evaluate_commonsenseqa_accuracy(model, tokenizer, split=split, max_examples=max_examples)
    raise ValueError(f"unsupported eval dataset_name={dataset_name!r}")


def start_fixed_curve_monitor(trainer, prefix: str, poll_seconds: float = 1.0):
    """
    跨 round 连续曲线监控器
        读取 trainer.state.log_history 中的 loss 记录
        用自增 fixed_step 作为横轴，构造 prefix/global_step 与 prefix/loss
        不依赖 trainer 的 step 字段，避免 FedMKT 每轮重置导致的覆盖问题
    """
    import threading
    import time

    stop_event = threading.Event()

    def pick_loss(rec: dict):
        for k in ("loss", "train/loss", "train_loss"):
            if k in rec:
                try:
                    return float(rec[k])
                except Exception:
                    return None
        return None

    def worker():
        last_seen = 0
        fixed_step = 0
        defined = False
        prev_hist_len = None

        while not stop_event.is_set():
            state = getattr(trainer, "state", None)
            hist = getattr(state, "log_history", None) if state is not None else None
            if not hist:
                time.sleep(poll_seconds)
                continue

            hist_len = len(hist)

            # log_history 被清空或替换：把读取指针拉回 0，fixed_step 不回退
            if prev_hist_len is not None and hist_len < last_seen:
                last_seen = 0
            prev_hist_len = hist_len

            # 定义 metric 的横轴绑定关系，只需要做一次
            if _tracking_active() and not defined:
                _tracking_define_metric(f"{prefix}/global_step")
                _tracking_define_metric(f"{prefix}/loss", step_metric=f"{prefix}/global_step")
                defined = True

            while last_seen < hist_len:
                rec = hist[last_seen]
                last_seen += 1
                if not isinstance(rec, dict):
                    continue

                loss = pick_loss(rec)
                if loss is None:
                    continue

                if _tracking_active():
                    _tracking_log(
                        {
                            f"{prefix}/global_step": int(fixed_step),
                            f"{prefix}/loss": float(loss),
                        }
                    )
                fixed_step += 1

            time.sleep(poll_seconds)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return stop_event


def train_llm(ctx):
    if PROJECT_PYTHON_DIR not in sys.path:
        sys.path.insert(0, PROJECT_PYTHON_DIR)

    ensure_torch_imported()
    from peft import LoraConfig, TaskType
    from fate_llm.model_zoo.pellm.llama import LLaMa
    from fate_llm.model_zoo.pellm.auto_causal_lm import AutoCausalLM
    from fate_llm.algo.fedmkt import FedMKTTrainingArguments, FedMKTLLM
    from fate_llm.dataset.qa_dataset import QaDataset
    from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
    from transformers import AutoConfig

    # 强制 init wandb，确保本进程的 wandb.run 可用
    ensure_wandb_run(ctx, name=f"llm_party{ctx.local.party[1]}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=llm_lora_target_modules,
    )

    llm_config = AutoConfig.from_pretrained(llm_pretrained_path)
    llm_model_class = LLaMa if getattr(llm_config, "model_type", None) == "llama" else AutoCausalLM

    model = llm_model_class(
        pretrained_path=llm_pretrained_path,
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype=llm_torch_dtype,
        model_load_kwargs=llm_model_load_kwargs,
    )
    print(f"[LLM device_map] {getattr(model, 'hf_device_map', None)}", flush=True)
    print_model_device("after_llm_load", model)
    print_cuda_memory("after_llm_load")

    pub_data = QaDataset(
        tokenizer_name_or_path=llm_pretrained_path,
        dataset_name=fedmkt_dataset_name,
        data_part="common",
        seq_max_len=512,
        select_num=public_select_num,
        start_idx=public_start_idx,
        need_preprocess=True,
    )
    pub_data.load(process_data_output_dir)
    print(
        f"[LLM data] dataset={fedmkt_dataset_name} data_dir={process_data_output_dir} "
        f"public_samples={len(pub_data)}",
        flush=True,
    )

    training_args = FedMKTTrainingArguments(
        global_epochs=global_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=batch_size,
        learning_rate=llm_lr,
        output_dir=training_output_dir,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        use_cpu=use_cpu,
        vocab_size=llm_config.vocab_size,
        kd_alpha=kd_alpha,
        distill_loss_type=distill_loss_type,
        distill_strategy=distill_strategy,
        distill_temperature=distill_temperature,
        report_to=[],
        logging_strategy="no",
        use_mmlcc_aggregation=use_mmlcc_aggregation,
        mmlcc_probability_epsilon=mmlcc_probability_epsilon,
        mmlcc_num_blocks=mmlcc_num_blocks,
        mmlcc_privacy_guarantee=mmlcc_privacy_guarantee,
        mmlcc_beta_radius=mmlcc_beta_radius,
        mmlcc_noise_sigma=mmlcc_noise_sigma,
        mmlcc_noise_clip_theta=mmlcc_noise_clip_theta,
        mmlcc_seed=mmlcc_seed,
    )

    slm_to_llm_vocab_mapping = []
    for path in slm_to_llm_vocab_mapping_paths:
        with open(path, "r") as fin:
            slm_to_llm_vocab_mapping.append(json.loads(fin.read()))

    slm_tokenizers = [get_tokenizer(p) for p in slm_pretrained_paths]
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

    print_cuda_memory("before_llm_train")
    trainer.train()
    print_cuda_memory("after_llm_train")

    task_metrics = evaluate_task_accuracy(
        model,
        tokenizer,
        fedmkt_dataset_name,
        split=eval_split,
        max_examples=eval_max_examples,
    )
    print(f"[LLM {fedmkt_dataset_name}] {task_metrics}")

    if _tracking_active():
        _tracking_log({f"llm_final/{k}": v for k, v in task_metrics.items()})

    trainer.save_model(llm_model_saved_directory)



def attach_noise_metadata(tokenized_qa_dataset, data_dir, data_part):
    """Reattach synthetic-noise metadata after QaDataset tokenization.

    QaDataset removes raw ARC columns during tokenization; GMMNoiseTrainer only needs
    numeric metadata for metrics, so we copy those columns back from the noisy split.
    """
    try:
        from datasets import load_from_disk
        raw = load_from_disk(data_dir)[data_part]
    except Exception as exc:
        print(f"[GMM metadata] skip: cannot load raw split {data_part} from {data_dir}: {exc}", flush=True)
        return tokenized_qa_dataset

    if len(raw) != len(tokenized_qa_dataset):
        print(
            f"[GMM metadata] skip: raw/tokenized length mismatch for {data_part}: "
            f"raw={len(raw)} tokenized={len(tokenized_qa_dataset)}",
            flush=True,
        )
        return tokenized_qa_dataset

    for col, default in (("is_noisy", 0), ("clean_idx", -1), ("noisy_idx", -1), ("train_idx", -1)):
        if col in raw.column_names and col not in tokenized_qa_dataset.column_names:
            tokenized_qa_dataset = tokenized_qa_dataset.add_column(col, [int(x) for x in raw[col]])
        elif col not in tokenized_qa_dataset.column_names:
            tokenized_qa_dataset = tokenized_qa_dataset.add_column(col, [default] * len(tokenized_qa_dataset))

    if "is_noisy" in tokenized_qa_dataset.column_names:
        noise_rate = sum(int(x) for x in tokenized_qa_dataset["is_noisy"]) / max(1, len(tokenized_qa_dataset))
        print(f"[GMM metadata] attached split={data_part} samples={len(tokenized_qa_dataset)} noise_rate={noise_rate:.6f}", flush=True)
    return tokenized_qa_dataset

def train_slm(ctx, slm_idx):
    if PROJECT_PYTHON_DIR not in sys.path:
        sys.path.insert(0, PROJECT_PYTHON_DIR)

    ensure_torch_imported()
    import transformers
    from peft import LoraConfig, TaskType
    from fate_llm.model_zoo.pellm.llama import LLaMa
    from fate_llm.model_zoo.pellm.gpt2 import GPT2CLM
    from fate_llm.model_zoo.pellm.opt import OPT
    from fate_llm.model_zoo.pellm.bloom import Bloom
    from fate_llm.algo.fedmkt_gmm import FedMKTGMMTrainingArguments, FedMKTGMMSLM
    from fate_llm.dataset.qa_dataset import QaDataset
    from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
    from transformers import AutoConfig

    ensure_wandb_run(ctx, name=f"slm{slm_idx}_party{ctx.local.party[1]}")

    slm_model_class = [OPT, GPT2CLM, LLaMa, Bloom]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=slm_lora_target_modules[slm_idx],
    )

    model = slm_model_class[slm_idx](
        pretrained_path=slm_pretrained_paths[slm_idx],
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype=slm_torch_dtype,
    )
    print_model_device(f"after_slm{slm_idx}_load", model)
    print_cuda_memory(f"after_slm{slm_idx}_load")

    priv_data = QaDataset(
        tokenizer_name_or_path=slm_pretrained_paths[slm_idx],
        dataset_name=fedmkt_dataset_name,
        data_part=f"client_{slm_idx}",
        seq_max_len=512,
        need_preprocess=True,
    )
    priv_data.load(process_data_output_dir)
    priv_data.ds = attach_noise_metadata(priv_data.ds, process_data_output_dir, f"client_{slm_idx}")

    pub_data = QaDataset(
        tokenizer_name_or_path=slm_pretrained_paths[slm_idx],
        dataset_name=fedmkt_dataset_name,
        data_part="common",
        seq_max_len=512,
        select_num=public_select_num,
        start_idx=public_start_idx,
        need_preprocess=True,
    )
    pub_data.load(process_data_output_dir)
    print(
        f"[SLM{slm_idx} data] dataset={fedmkt_dataset_name} data_dir={process_data_output_dir} "
        f"private_samples={len(priv_data)} public_samples={len(pub_data)}",
        flush=True,
    )

    training_args = FedMKTGMMTrainingArguments(
        global_epochs=global_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=batch_size,
        learning_rate=slm_lrs[slm_idx],
        output_dir=training_output_dir,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        use_cpu=use_cpu,
        vocab_size=AutoConfig.from_pretrained(slm_pretrained_paths[slm_idx]).vocab_size,
        kd_alpha=kd_alpha,
        distill_loss_type=distill_loss_type,
        distill_strategy=distill_strategy,
        distill_temperature=distill_temperature,
        report_to=[],
        logging_strategy="no",
        gmm_enabled=gmm_enabled,
        gmm_warmup_steps=gmm_warmup_steps,
        gmm_update_interval=gmm_update_interval,
        gmm_history_size=gmm_history_size,
        gmm_min_samples=gmm_min_samples,
        gmm_clean_threshold=gmm_clean_threshold,
        gmm_noisy_weight=gmm_noisy_weight,
        gmm_weight_power=gmm_weight_power,
        denoise_loss_type=denoise_loss_type,
        sce_alpha=sce_alpha,
        sce_beta=sce_beta,
        rce_epsilon=rce_epsilon,
        denoise_loss_log_every_n_steps=denoise_loss_log_every_n_steps,
    )
    force_training_args_device(training_args)

    tokenizer = get_tokenizer(slm_pretrained_paths[slm_idx])

    with open(llm_to_slm_vocab_mapping_paths[slm_idx], "r") as fin:
        vocab_mapping = json.loads(fin.read())

    trainer = FedMKTGMMSLM(
        ctx=ctx,
        model=model,
        training_args=training_args,
        pub_train_set=pub_data,
        priv_train_set=priv_data,
        tokenizer=tokenizer,
        save_trainable_weights_only=True,
        llm_tokenizer=get_tokenizer(llm_pretrained_path),
        llm_to_slm_vocab_mapping=vocab_mapping,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer),
    )

    trainer.train()

    task_metrics = evaluate_task_accuracy(
        model,
        tokenizer,
        fedmkt_dataset_name,
        split=eval_split,
        max_examples=eval_max_examples,
    )
    print(f"[SLM{slm_idx} {fedmkt_dataset_name}] {task_metrics}")

    if _tracking_active():
        _tracking_log({f"slm{slm_idx}_final/{k}": v for k, v in task_metrics.items()})

    trainer.save_model(slm_models_saved_directory[slm_idx])


def run(ctx):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[path] PROJECT_PYTHON_DIR={PROJECT_PYTHON_DIR}", flush=True)
    print(
        f"[config] path={experiment_config_path} task={fedmkt_active_task} dataset={fedmkt_dataset_name} "
        f"data_dir={process_data_output_dir} mmlcc={use_mmlcc_aggregation} gmm={gmm_enabled} denoise_loss={denoise_loss_type} "
        f"wandb_mode={wandb_mode} swanlab_mode={swanlab_mode} "
        f"arbiter_cuda={arbiter_cuda_visible_devices} llm_dtype={llm_torch_dtype} "
        f"llm_device_map={llm_model_load_kwargs.get('device_map')}",
        flush=True,
    )
    if ctx.is_on_arbiter:
        configure_process_cuda(ctx, "arbiter_before_train_llm", arbiter_cuda_visible_devices)
        train_llm(ctx)
    elif ctx.is_on_guest:
        configure_process_cuda(ctx, "guest_before_train_slm0", guest_cuda_visible_devices, force_single_device=True)
        train_slm(ctx, slm_idx=0)
    else:
        if ctx.local.party[1] == "9999":
            host_visible_devices = str(host_cuda_visible_devices.get("9999", "5"))
            slm_idx = 1
        elif ctx.local.party[1] == "10000":
            host_visible_devices = str(host_cuda_visible_devices.get("10000", "6"))
            slm_idx = 2
        elif ctx.local.party[1] == "10001":
            host_visible_devices = str(host_cuda_visible_devices.get("10001", "7"))
            slm_idx = 3
        else:
            raise ValueError(f"party_id={ctx.local.party[1]} is illegal")

        configure_process_cuda(ctx, f"host_before_train_slm{slm_idx}", host_visible_devices, force_single_device=True)
        train_slm(ctx, slm_idx=slm_idx)


if __name__ == "__main__":
    launch(run)
