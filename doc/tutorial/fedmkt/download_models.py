import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise ImportError(
        "Missing dependency: huggingface_hub\n"
        "Please install it first:\n"
        "  pip install -U huggingface_hub"
    ) from exc


MODELS: List[Dict[str, str]] = [
    {
        "name": "Qwen2.5-14B-Instruct",
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "local_dir": "/home/cmcc/lhy/models/Qwen2.5-14B-Instruct",
    },
    {
        "name": "opt-1.3b",
        "repo_id": "facebook/opt-1.3b",
        "local_dir": "/home/cmcc/lhy/models/opt-1.3b",
    },
    {
        "name": "gpt2-xl",
        "repo_id": "openai-community/gpt2-xl",
        "local_dir": "/home/cmcc/lhy/models/gpt2-xl",
    },
    {
        "name": "Sheared-LLaMA-1.3B",
        "repo_id": "princeton-nlp/Sheared-LLaMA-1.3B",
        "local_dir": "/home/cmcc/lhy/models/Sheared-LLaMA-1.3B",
    },
    {
        "name": "bloom-1b1",
        "repo_id": "bigscience/bloom-1b1",
        "local_dir": "/home/cmcc/lhy/models/bloom-1b1",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FedMKT LLM/SLM models from HuggingFace."
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", None),
        help="Optional HuggingFace cache directory. Default: HF_HOME or huggingface_hub default.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision, branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Optional HuggingFace token. You can also set HF_TOKEN in environment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing local model directory before downloading.",
    )
    parser.add_argument(
        "--local-dir-use-symlinks",
        type=str,
        default="False",
        choices=["True", "False", "auto"],
        help=(
            "Passed to snapshot_download. Use False to put real files in local_dir. "
            "Default: False."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip download if local directory already looks complete. Default: enabled.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Always call snapshot_download even if local files already exist.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=(
            "Comma-separated model names to download. "
            "Example: --only Qwen2.5-14B-Instruct,opt-1.3b"
        ),
    )

    return parser.parse_args()


def looks_like_hf_model_dir(path: Path) -> bool:
    """
    A loose check for a completed HuggingFace model directory.
    It avoids re-downloading when config/tokenizer/weights are already present.
    """
    if not path.exists() or not path.is_dir():
        return False

    has_config = (path / "config.json").exists()
    has_tokenizer = any(
        (path / name).exists()
        for name in [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "spiece.model",
        ]
    )
    has_weights = any(
        path.glob(pattern)
        for pattern in [
            "*.safetensors",
            "pytorch_model*.bin",
            "model*.safetensors",
            "*.bin",
        ]
    )

    return has_config and has_weights and has_tokenizer


def normalize_symlink_arg(value: str):
    if value == "True":
        return True
    if value == "False":
        return False
    return "auto"


def download_one(model: Dict[str, str], args: argparse.Namespace) -> None:
    name = model["name"]
    repo_id = model["repo_id"]
    local_dir = Path(model["local_dir"])

    print("=" * 100)
    print(f"[model] {name}")
    print(f"[repo]  {repo_id}")
    print(f"[dir]   {local_dir}")

    if args.force and local_dir.exists():
        print(f"[force] removing existing directory: {local_dir}")
        shutil.rmtree(local_dir)

    if args.skip_existing and looks_like_hf_model_dir(local_dir):
        print(f"[skip] local directory already looks complete: {local_dir}")
        return

    local_dir.parent.mkdir(parents=True, exist_ok=True)

    print("[download] starting snapshot_download ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=normalize_symlink_arg(args.local_dir_use_symlinks),
        cache_dir=args.cache_dir,
        revision=args.revision,
        token=args.token,
        resume_download=True,
        ignore_patterns=[
            "*.msgpack",
            "*.h5",
            "*.ot",
            "*.onnx",
            "*.tflite",
            "*.gguf",
        ],
    )

    if looks_like_hf_model_dir(local_dir):
        print(f"[ok] downloaded successfully: {local_dir}")
    else:
        print(
            f"[warning] download finished, but directory may be incomplete: {local_dir}\n"
            "Please check network errors or HuggingFace access permissions."
        )


def main() -> None:
    args = parse_args()

    selected = MODELS
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [m for m in MODELS if m["name"] in wanted or m["repo_id"] in wanted]
        missing = wanted - {m["name"] for m in selected} - {m["repo_id"] for m in selected}
        if missing:
            print(f"[error] unknown model name/repo_id in --only: {sorted(missing)}")
            print("Available names:")
            for m in MODELS:
                print(f"  - {m['name']}  ({m['repo_id']})")
            sys.exit(1)

    print("[info] models to download:")
    for m in selected:
        print(f"  - {m['repo_id']} -> {m['local_dir']}")

    failed = []

    for model in selected:
        try:
            download_one(model, args)
        except Exception as exc:
            failed.append((model["name"], str(exc)))
            print(f"[failed] {model['name']}: {exc}")

    print("=" * 100)
    if failed:
        print("[summary] some models failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)

    print("[summary] all selected models downloaded successfully.")


if __name__ == "__main__":
    main()