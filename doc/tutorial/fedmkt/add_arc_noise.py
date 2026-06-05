#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a noisy ARC-C FedMKT dataset.

The script reads the clean FedMKT dataset directory configured in
``configs/default.yaml`` (``data.tasks.arc_c.data_dir`` by default), flips ARC
``answerKey`` labels for selected training client splits, and saves a new
DatasetDict to ``noise.output_dir`` / ``data.tasks.arc_c.noisy_data_dir``.

The noisy dataset keeps the normal ARC columns, so existing ``QaDataset`` and
FedMKT preprocessing still work.  It also adds metadata columns used only for
analysis and GMM denoising metrics:
``answerKey_clean``, ``answerKey_noisy``, ``clean_idx``, ``noisy_idx``,
``is_noisy``, ``noise_source`` and ``noise_seed``.
"""

import argparse
import ast
import json
import os
import random
from typing import Dict, List, Optional

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

DATASET_REGISTRY = {
    "arc_challenge": ("ai2_arc", "ARC-Challenge"),
    "arc_c": ("ai2_arc", "ARC-Challenge"),
    "arc_easy": ("ai2_arc", "ARC-Easy"),
    "arc_e": ("ai2_arc", "ARC-Easy"),
}


def _parse_scalar(value):
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


def _strip_comment(line):
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
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if line.strip():
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
            parent.append(_parse_scalar(content[2:]))
            continue
        if ":" not in content:
            raise ValueError(f"unsupported YAML line: {content}")
        key, value = content.split(":", 1)
        key, value = key.strip(), value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue
        next_is_list = idx + 1 < len(cleaned) and cleaned[idx + 1][0] > indent and cleaned[idx + 1][1].startswith("- ")
        container = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))
    return root


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f) or {}
        text = f.read()
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return _load_simple_yaml(text)


def cfg(config, path, default=None):
    cur = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def canonical_task(name):
    aliases = {"arc_c": "arc_c", "arc_challenge": "arc_c", "arc_e": "arc_e", "arc_easy": "arc_e"}
    return aliases.get(str(name).strip().lower().replace("-", "_"), str(name).strip().lower())


def canonical_dataset(name):
    aliases = {"arc_c": "arc_challenge", "arc_challenge": "arc_challenge", "arc_e": "arc_easy", "arc_easy": "arc_easy"}
    return aliases.get(str(name).strip().lower().replace("-", "_"), str(name).strip().lower())


def normalize_answer_index(answer_key, labels: List[str], texts: List[str]) -> Optional[int]:
    key = str(answer_key).strip()
    labels = [str(x).strip() for x in labels]
    for i, lab in enumerate(labels):
        if key == lab:
            return i
    key_up = key.upper()
    if key_up in LETTERS and LETTERS.index(key_up) < len(texts):
        return LETTERS.index(key_up)
    if key.isdigit():
        idx = int(key) - 1
        if 0 <= idx < len(texts):
            return idx
    for i, text in enumerate(texts):
        if key.lower() == str(text).strip().lower():
            return i
    return None


def answer_key_for_idx(idx: int, labels: List[str], prefer_original_labels: bool = True) -> str:
    if prefer_original_labels and 0 <= idx < len(labels):
        return str(labels[idx])
    return LETTERS[idx]


def inject_noise_to_split(split, noise_rate: float, noise_type: str, seed: int, split_name: str):
    rng = random.Random(seed)
    noisy_count = 0
    clean_label_count: Dict[str, int] = {}
    noisy_label_count: Dict[str, int] = {}

    def add_noise(ex, idx):
        nonlocal noisy_count
        choices = ex.get("choices", {})
        texts = [str(x) for x in choices.get("text", [])]
        labels = [str(x) for x in choices.get("label", [])]
        if len(labels) != len(texts):
            labels = LETTERS[: len(texts)]
        clean_idx = normalize_answer_index(ex.get("answerKey"), labels, texts)
        if clean_idx is None or len(texts) < 2:
            clean_idx = 0
        clean_key = answer_key_for_idx(clean_idx, labels)
        clean_label_count[clean_key] = clean_label_count.get(clean_key, 0) + 1

        if rng.random() < noise_rate:
            if noise_type == "sym":
                candidates = [i for i in range(len(texts)) if i != clean_idx]
                noisy_idx = rng.choice(candidates)
            elif noise_type == "asym":
                noisy_idx = (clean_idx + 1) % len(texts)
            else:
                raise ValueError(f"unsupported noise_type={noise_type!r}")
            is_noisy = 1
            noisy_count += 1
        else:
            noisy_idx = clean_idx
            is_noisy = 0
        noisy_key = answer_key_for_idx(noisy_idx, labels)
        noisy_label_count[noisy_key] = noisy_label_count.get(noisy_key, 0) + 1

        ex["answerKey_clean"] = clean_key
        ex["answerKey_noisy"] = noisy_key
        ex["answerKey"] = noisy_key
        ex["clean_idx"] = int(clean_idx)
        ex["noisy_idx"] = int(noisy_idx)
        ex["train_idx"] = int(noisy_idx)
        ex["is_noisy"] = int(is_noisy)
        ex["noise_source"] = split_name
        ex["noise_seed"] = int(seed)
        return ex

    out = split.map(add_noise, with_indices=True, load_from_cache_file=False)
    stats = {
        "split": split_name,
        "noise_type": noise_type,
        "target_noise_rate": noise_rate,
        "actual_noise_rate": noisy_count / max(1, len(split)),
        "num_noisy_samples": noisy_count,
        "num_samples": len(split),
        "clean_label_count": clean_label_count,
        "noisy_label_count": noisy_label_count,
        "seed": seed,
    }
    return out, stats


def mark_clean_split(split, split_name: str):
    def mark(ex):
        choices = ex.get("choices", {})
        texts = [str(x) for x in choices.get("text", [])]
        labels = [str(x) for x in choices.get("label", [])]
        if len(labels) != len(texts):
            labels = LETTERS[: len(texts)]
        clean_idx = normalize_answer_index(ex.get("answerKey"), labels, texts) or 0
        clean_key = answer_key_for_idx(clean_idx, labels)
        ex["answerKey_clean"] = clean_key
        ex["answerKey_noisy"] = clean_key
        ex["clean_idx"] = int(clean_idx)
        ex["noisy_idx"] = int(clean_idx)
        ex["train_idx"] = int(clean_idx)
        ex["is_noisy"] = 0
        ex["noise_source"] = f"{split_name}:clean"
        ex["noise_seed"] = -1
        return ex
    return split.map(mark, load_from_cache_file=False)


def load_or_prepare_clean_dataset(input_dir, dataset_name, client_num, source_split, seed):
    from datasets import DatasetDict, load_dataset, load_from_disk
    if input_dir and os.path.exists(input_dir):
        return load_from_disk(input_dir)
    path, subset = DATASET_REGISTRY[canonical_dataset(dataset_name)]
    raw = load_dataset(path, subset)
    train = raw[source_split].shuffle(seed=seed)
    part_size = len(train) // (client_num + 1)
    if part_size <= 0:
        raise ValueError(f"not enough rows: {len(train)} for client_num={client_num}")
    output = {}
    for i in range(client_num):
        output[f"client_{i}"] = train.select(range(i * part_size, (i + 1) * part_size))
    output["common"] = train.select(range(client_num * part_size, (client_num + 1) * part_size))
    for split_name in ("validation", "test"):
        if split_name in raw:
            output[split_name] = raw[split_name]
    return DatasetDict(output)


def parse_parts(value: str, client_num: int):
    value = str(value or "clients").strip().lower()
    if value in {"client", "clients", "private"}:
        return {f"client_{i}" for i in range(client_num)}
    if value in {"all_train", "train"}:
        return {f"client_{i}" for i in range(client_num)} | {"common"}
    if value in {"all"}:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def main():
    parser = argparse.ArgumentParser(description="Add synthetic label noise to ARC-C FedMKT data.")
    parser.add_argument("--config", default=os.environ.get("FEDMKT_CONFIG", os.path.join(os.path.dirname(__file__), "configs", "default.yaml")))
    parser.add_argument("--task", default=os.environ.get("FEDMKT_TASK"))
    parser.add_argument("--input_dir", default=os.environ.get("FEDMKT_CLEAN_DATA_DIR"))
    parser.add_argument("--output_dir", default=os.environ.get("FEDMKT_NOISY_DATA_DIR"))
    parser.add_argument("--noise_rate", type=float, default=None)
    parser.add_argument("--noise_type", choices=["sym", "asym"], default=None)
    parser.add_argument("--noise_parts", default=None, help="clients/private, all_train, all, or comma list like client_0,client_1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--client_num", type=int, default=None)
    parser.add_argument("--source_split", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    task = canonical_task(args.task or cfg(config, "data.active_task", "arc_c"))
    task_cfg = cfg(config, f"data.tasks.{task}", {}) or {}
    dataset_name = canonical_dataset(task_cfg.get("dataset_name", "arc_challenge"))
    input_dir = args.input_dir or task_cfg.get("data_dir") or cfg(config, "data.data_dir")
    output_dir = args.output_dir or task_cfg.get("noisy_data_dir") or cfg(config, "noise.output_dir")
    noise_rate = float(args.noise_rate if args.noise_rate is not None else cfg(config, "noise.noise_rate", 0.4))
    noise_type = args.noise_type or cfg(config, "noise.noise_type", "sym")
    seed = int(args.seed if args.seed is not None else cfg(config, "noise.seed", cfg(config, "data_prepare.seed", 42)))
    client_num = int(args.client_num if args.client_num is not None else cfg(config, "data_prepare.client_num", 4))
    source_split = args.source_split or cfg(config, "data_prepare.source_split", "train")
    noise_parts = parse_parts(args.noise_parts or cfg(config, "noise.noise_parts", "clients"), client_num)

    if task not in {"arc_c", "arc_e"}:
        raise ValueError(f"add_arc_noise.py currently supports ARC tasks only, got task={task!r}")
    if not output_dir:
        base = input_dir.rstrip(os.sep) if input_dir else os.path.join(os.getcwd(), "data", task)
        output_dir = f"{base}_noise_{noise_type}_{noise_rate:g}_seed{seed}"

    ds = load_or_prepare_clean_dataset(input_dir, dataset_name, client_num, source_split, seed)
    noisy_splits = {}
    stats = {"task": task, "dataset_name": dataset_name, "input_dir": input_dir, "output_dir": output_dir, "splits": {}}
    for split_name, split in ds.items():
        should_noise = (noise_parts is None) or (split_name in noise_parts)
        if should_noise:
            noisy_split, split_stats = inject_noise_to_split(split, noise_rate, noise_type, seed + abs(hash(split_name)) % 100000, split_name)
        else:
            noisy_split = mark_clean_split(split, split_name)
            split_stats = {"split": split_name, "actual_noise_rate": 0.0, "num_noisy_samples": 0, "num_samples": len(split), "noise_type": "clean"}
        noisy_splits[split_name] = noisy_split
        stats["splits"][split_name] = split_stats

    from datasets import DatasetDict
    output = DatasetDict(noisy_splits)
    os.makedirs(output_dir, exist_ok=True)
    output.save_to_disk(output_dir)
    with open(os.path.join(output_dir, "noise_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] noisy dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
