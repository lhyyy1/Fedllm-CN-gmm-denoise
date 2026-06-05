import csv
import json
import os
import re
import time
from datetime import datetime, timezone


_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_VALUES = {"", "0", "false", "no", "n", "off", "disable", "disabled", "none"}


def _as_bool(value, default=False):
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _safe_name(value):
    value = str(value or "run").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "run"


def _normalize(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return _normalize(value.detach().cpu().item())
            return value.detach().cpu().tolist()
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def configure_local_tracking(
    enabled=None,
    log_dir=None,
    run_name=None,
    write_jsonl=None,
    write_csv=None,
):
    if enabled is not None:
        os.environ["FEDMKT_LOCAL_TRACKING_ENABLED"] = "true" if bool(enabled) else "false"
    if log_dir:
        os.environ["FEDMKT_LOCAL_LOG_DIR"] = str(log_dir)
    if run_name:
        os.environ["FEDMKT_LOCAL_RUN_NAME"] = _safe_name(run_name)
    if write_jsonl is not None:
        os.environ["FEDMKT_LOCAL_WRITE_JSONL"] = "true" if bool(write_jsonl) else "false"
    if write_csv is not None:
        os.environ["FEDMKT_LOCAL_WRITE_CSV"] = "true" if bool(write_csv) else "false"


def set_local_run_name(run_name):
    if run_name:
        os.environ["FEDMKT_LOCAL_RUN_NAME"] = _safe_name(run_name)


def local_tracking_enabled():
    return _as_bool(os.environ.get("FEDMKT_LOCAL_TRACKING_ENABLED"), default=False)


def _log_paths():
    log_dir = os.environ.get("FEDMKT_LOCAL_LOG_DIR", "local_metrics")
    os.makedirs(log_dir, exist_ok=True)

    run_name = _safe_name(os.environ.get("FEDMKT_LOCAL_RUN_NAME", "run"))
    pid = os.getpid()
    stem = f"metrics_{run_name}_pid{pid}"
    return os.path.join(log_dir, stem + ".jsonl"), os.path.join(log_dir, stem + ".csv")


def _write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path, record):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "timestamp",
                "unix_time",
                "pid",
                "run_name",
                "metric",
                "value",
            ],
        )
        if not exists:
            writer.writeheader()

        for metric, value in record["metrics"].items():
            writer.writerow(
                {
                    "timestamp": record["timestamp"],
                    "unix_time": record["unix_time"],
                    "pid": record["pid"],
                    "run_name": record["run_name"],
                    "metric": metric,
                    "value": json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value,
                }
            )


def log_local_metrics(metrics):
    if not local_tracking_enabled() or not metrics:
        return

    normalized_metrics = _normalize(metrics)
    now = time.time()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unix_time": now,
        "pid": os.getpid(),
        "run_name": _safe_name(os.environ.get("FEDMKT_LOCAL_RUN_NAME", "run")),
        "metrics": normalized_metrics,
    }

    jsonl_path, csv_path = _log_paths()
    if _as_bool(os.environ.get("FEDMKT_LOCAL_WRITE_JSONL"), default=True):
        _write_jsonl(jsonl_path, record)
    if _as_bool(os.environ.get("FEDMKT_LOCAL_WRITE_CSV"), default=True):
        _write_csv(csv_path, record)
