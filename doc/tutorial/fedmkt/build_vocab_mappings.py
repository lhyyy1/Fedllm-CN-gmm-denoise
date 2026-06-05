import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PROJECT_PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PROJECT_PYTHON_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PYTHON_DIR)

from fate_llm.algo.fedmkt.token_alignment.vocab_mapping import get_vocab_mappings


def load_config(config_path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the FedMKT config file.") from exc

    with open(config_path, "r", encoding="utf-8") as fin:
        data = yaml.safe_load(fin) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid config at {config_path}")
    return data


def ensure_list(value, field_name):
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def resolve_path(base_dir, path_value):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def main():
    parser = argparse.ArgumentParser(description="Generate FedMKT vocab mappings for the current config.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
        help="FedMKT config file path",
    )
    parser.add_argument(
        "--num-processors",
        type=int,
        default=8,
        help="worker processes used for edit-distance based mapping",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing mapping files",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})

    llm_pretrained = paths.get("llm_pretrained")
    slm_pretrained = ensure_list(paths.get("slm_pretrained", []), "paths.slm_pretrained")
    mapping_dir = paths.get("vocab_mapping_dir", "./mapping")
    slm_to_llm = ensure_list(paths.get("slm_to_llm_vocab_mappings", []), "paths.slm_to_llm_vocab_mappings")
    llm_to_slm = ensure_list(paths.get("llm_to_slm_vocab_mappings", []), "paths.llm_to_slm_vocab_mappings")

    if not llm_pretrained:
        raise ValueError("paths.llm_pretrained is required")
    if not (len(slm_pretrained) == len(slm_to_llm) == len(llm_to_slm)):
        raise ValueError("slm_pretrained, slm_to_llm_vocab_mappings and llm_to_slm_vocab_mappings must have equal length")

    fedmkt_dir = os.path.dirname(os.path.abspath(__file__))
    llm_pretrained = resolve_path(fedmkt_dir, llm_pretrained)
    slm_pretrained = [resolve_path(fedmkt_dir, path) for path in slm_pretrained]
    mapping_dir_abs = resolve_path(fedmkt_dir, mapping_dir)
    os.makedirs(mapping_dir_abs, exist_ok=True)

    for idx, slm_path in enumerate(slm_pretrained):
        pairs = [
            (slm_path, llm_pretrained, slm_to_llm[idx]),
            (llm_pretrained, slm_path, llm_to_slm[idx]),
        ]
        for src_model, dst_model, mapping_name in pairs:
            mapping_path = os.path.join(mapping_dir_abs, mapping_name)
            if os.path.exists(mapping_path) and not args.force:
                print(f"[skip] {mapping_path}")
                continue

            print(f"[build] {src_model} -> {dst_model} => {mapping_path}", flush=True)
            get_vocab_mappings(
                src_model,
                dst_model,
                mapping_path,
                num_processors=args.num_processors,
            )

    print("[done] vocab mappings are ready", flush=True)


if __name__ == "__main__":
    main()
