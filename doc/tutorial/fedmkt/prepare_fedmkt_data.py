import argparse
import ast
import json
import os


DATASET_REGISTRY = {
    "arc_challenge": ("ai2_arc", "ARC-Challenge"),
    "arc_c": ("ai2_arc", "ARC-Challenge"),
    "arc_easy": ("ai2_arc", "ARC-Easy"),
    "arc_e": ("ai2_arc", "ARC-Easy"),
    "boolq": ("super_glue", "boolq"),
    "bool_q": ("super_glue", "boolq"),
    "cqa": ("commonsense_qa", None),
    "commonsenseqa": ("commonsense_qa", None),
    "commonsense_qa": ("commonsense_qa", None),
    "rte": ("glue", "rte"),
}


def normalize_dataset_name(dataset_name):
    return dataset_name.strip().lower().replace("-", "_")


def canonical_dataset_name(dataset_name):
    aliases = {
        "arc_c": "arc_challenge",
        "arc_challenge": "arc_challenge",
        "arc_e": "arc_easy",
        "arc_easy": "arc_easy",
        "boolq": "boolq",
        "bool_q": "boolq",
        "cqa": "commonsenseqa",
        "commonsenseqa": "commonsenseqa",
        "commonsense_qa": "commonsenseqa",
        "rte": "rte",
    }
    normalized = normalize_dataset_name(dataset_name)
    return aliases.get(normalized, normalized)


def canonical_task_name(task_name):
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
    normalized = normalize_dataset_name(task_name)
    return aliases.get(normalized, normalized)


def parse_simple_yaml_scalar(value):
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


def strip_yaml_comment(line):
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


def load_simple_yaml(text):
    cleaned = []
    for raw_line in text.splitlines():
        line = strip_yaml_comment(raw_line).rstrip()
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
            parent.append(parse_simple_yaml_scalar(content[2:]))
            continue

        if ":" not in content:
            raise ValueError(f"unsupported YAML line: {content}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            parent[key] = parse_simple_yaml_scalar(value)
            continue

        next_is_list = False
        if idx + 1 < len(cleaned):
            next_indent, next_content = cleaned[idx + 1]
            next_is_list = next_indent > indent and next_content.startswith("- ")
        container = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))

    return root


def load_experiment_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fin:
        if path.endswith(".json"):
            return json.load(fin) or {}
        text = fin.read()
        try:
            import yaml
        except ImportError:
            return load_simple_yaml(text)
        return yaml.safe_load(text) or {}


def cfg(config, path, default=None):
    cur = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pick_arg(cli_value, config, path, default=None, cast=None):
    value = cli_value
    if value is None:
        value = cfg(config, path, default)
    if cast is not None and value is not None:
        return cast(value)
    return value


def resolve_data_settings(args, config):
    task = canonical_task_name(pick_arg(args.task, config, "data.active_task", "arc_c"))
    task_config = cfg(config, f"data.tasks.{task}", {})

    dataset_name = args.dataset_name
    if dataset_name is None and isinstance(task_config, dict):
        dataset_name = task_config.get("dataset_name")
    if dataset_name is None:
        dataset_name = cfg(config, "data.dataset_name", "rte")

    output_dir = args.output_dir
    if output_dir is None and isinstance(task_config, dict):
        output_dir = task_config.get("data_dir")
    if output_dir is None:
        output_dir = cfg(config, "data.data_dir", None)

    return task, canonical_dataset_name(dataset_name), output_dir


def load_source_dataset(dataset_name):
    from datasets import load_dataset

    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name in DATASET_REGISTRY:
        path, name = DATASET_REGISTRY[dataset_name]
        if name is None:
            return load_dataset(path)
        return load_dataset(path, name)
    return load_dataset(dataset_name)


def build_fedmkt_splits(raw_dataset, client_num, source_split, seed):
    from datasets import DatasetDict

    if source_split not in raw_dataset:
        raise ValueError(f"source_split={source_split!r} is not available in {list(raw_dataset.keys())}")

    train_data = raw_dataset[source_split].shuffle(seed=seed)
    part_size = len(train_data) // (client_num + 1)
    if part_size <= 0:
        raise ValueError(
            f"not enough rows in split={source_split!r}: rows={len(train_data)}, client_num={client_num}"
        )

    output = {}
    for client_idx in range(client_num):
        start = 0 # client_idx * part_size
        end = client_num * part_size # (client_idx + 1) * part_size
        output[f"client_{client_idx}"] = train_data.select(range(start, end))

    common_start = client_num * part_size
    common_end = (client_num + 1) * part_size
    output["common"] = train_data.select(range(common_start, common_end))

    for split_name in ("validation", "test"):
        if split_name in raw_dataset:
            output[split_name] = raw_dataset[split_name]

    return DatasetDict(output)


def main():
    parser = argparse.ArgumentParser(description="Prepare common/client_i splits for FedMKT QA experiments.")
    parser.add_argument(
        "--config",
        default=os.environ.get("FEDMKT_CONFIG", os.path.join(os.path.dirname(__file__), "configs", "default.yaml")),
        help="FedMKT YAML/JSON config. CLI arguments override config values.",
    )
    parser.add_argument(
        "--task",
        default=os.environ.get("FEDMKT_TASK"),
        help="Task key in data.tasks, for example: arc_c, arc_e, rte, boolq, cqa.",
    )
    parser.add_argument(
        "--dataset_name",
        default=os.environ.get("FEDMKT_DATASET_NAME"),
        help=(
            "One of: arc_challenge/arc_c, arc_easy/arc_e, boolq/bool_q, "
            "cqa/commonsenseqa/commonsense_qa, rte, or a HF dataset path."
        ),
    )
    parser.add_argument("--output_dir", default=os.environ.get("FEDMKT_DATA_DIR"), help="Directory passed to FEDMKT_DATA_DIR.")
    parser.add_argument("--client_num", type=int, default=None)
    parser.add_argument("--source_split", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    task, dataset_name, output_dir = resolve_data_settings(args, config)
    client_num = pick_arg(args.client_num, config, "data_prepare.client_num", 4, int)
    source_split = pick_arg(args.source_split, config, "data_prepare.source_split", "train")
    seed = pick_arg(args.seed, config, "data_prepare.seed", 42, int)
    if not output_dir:
        raise ValueError("output_dir is required. Set --output_dir or data.tasks.<task>.data_dir in the config.")

    raw_dataset = load_source_dataset(dataset_name)
    fedmkt_dataset = build_fedmkt_splits(
        raw_dataset=raw_dataset,
        client_num=client_num,
        source_split=source_split,
        seed=seed,
    )

    os.makedirs(output_dir, exist_ok=True)
    fedmkt_dataset.save_to_disk(output_dir)

    print(f"config={args.config}")
    print(f"saved task={task} dataset={dataset_name} source_split={source_split} client_num={client_num} seed={seed}")
    print(f"output_dir={output_dir}")
    for split_name, split_data in fedmkt_dataset.items():
        print(f"{split_name}: {len(split_data)}")


if __name__ == "__main__":
    main()
