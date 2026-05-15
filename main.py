import copy
import csv
import os.path as osp

import numpy as np

from manager import Manager
from utils import get_args, get_config, init, print_config


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def _numeric_keys(rows):
    keys = set()
    for row in rows:
        for key, value in row.items():
            if key == "fold":
                continue
            if isinstance(value, (int, float, np.floating)):
                keys.add(key)
    return sorted(keys)


def _mean_std(rows, keys):
    means, stds = {}, {}
    for key in keys:
        arr = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        arr = arr[~np.isnan(arr)]
        means[key] = float(np.nan) if arr.size == 0 else float(np.mean(arr))
        stds[key] = 0.0 if arr.size <= 1 else float(np.std(arr, ddof=1))
    return means, stds


def _write_fold_summary(path, rows):
    if not rows:
        return
    keys = _numeric_keys(rows)
    means, stds = _mean_std(rows, keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold"] + keys)
        for row in rows:
            writer.writerow([row["fold"]] + [f"{row.get(key, np.nan):.6f}" for key in keys])
        writer.writerow(["mean"] + [f"{means[key]:.6f}" for key in keys])
        writer.writerow(["std"] + [f"{stds[key]:.6f}" for key in keys])

    msg = ", ".join(f"{key}={means[key]:.6f}+/-{stds[key]:.6f}" for key in keys)
    print(f"[CV] Mean+/-Std -> {msg}")
    print(f"[CV] Saved summary to: {path}")


def _collect_task_cl_metrics(manager, fold_id, task_cl_rows):
    metrics = getattr(manager, "task_cl_metrics_per_task", {}) or {}
    keys = [
        "acc", "auc", "pr_auc",
        "forget_acc", "forget_auc", "forget_pr_auc",
    ]

    for dataset_name, split_dict in metrics.items():
        if not isinstance(split_dict, dict):
            split_dict = {"val": split_dict}
        for split_name, values in split_dict.items():
            if split_name not in ("val", "test") or not isinstance(values, dict):
                continue
            row = {"fold": fold_id}
            for key in keys:
                if key in values:
                    row[key] = float(values[key])
            task_cl_rows.setdefault(split_name, {}).setdefault(dataset_name, []).append(row)


def _write_task_cl_cv(task_cl_rows):
    for split_name, dataset_map in task_cl_rows.items():
        for dataset_name, rows in dataset_map.items():
            if not rows:
                continue
            out_csv = f"cv10_task_cl_{split_name}_{dataset_name}.csv"
            _write_fold_summary(out_csv, rows)


def _run_cv(config, cv_cfg):
    k_folds = int(cv_cfg.get("k_folds", 10))
    out_csv = cv_cfg.get("summary_csv", "cv10_results_kmp_mil.csv")
    fold_results = []
    task_cl_rows = {"val": {}, "test": {}}

    for fold_id in range(1, k_folds + 1):
        cfg_fold = copy.deepcopy(config)
        cfg_fold["cv"]["fold_id"] = fold_id
        cfg_fold["task_num"] = 0
        cfg_fold["save_result_dir"] = osp.join(
            cfg_fold["save_result_dir"], f"cv_fold_{fold_id:02d}"
        )

        print(f"\n========== [CV] Fold {fold_id}/{k_folds} ==========")
        manager = Manager(cfg_fold)
        result = manager.incre_train()

        if isinstance(result, dict):
            fold_results.append({"fold": fold_id, **result})
            shown = ", ".join(
                f"{key}={float(value):.6f}"
                for key, value in result.items()
                if isinstance(value, (int, float, np.floating))
            )
            print(f"[CV] Fold {fold_id:02d} result -> {shown}")

        _collect_task_cl_metrics(manager, fold_id, task_cl_rows)

    _write_fold_summary(out_csv, fold_results)
    _write_task_cl_cv(task_cl_rows)


def main(config):
    print_config(config)
    cv_cfg = config.get("cv", {}) or {}
    if _as_bool(cv_cfg.get("enabled", False)) and _as_bool(cv_cfg.get("run_all_folds", False)):
        _run_cv(config, cv_cfg)
    else:
        manager = Manager(config)
        manager.incre_train()
    print("[info] end of program execution")


if __name__ == "__main__":
    cfg = get_args()
    config = get_config(cfg["config"])
    init(cfg, config)
    main(config)
