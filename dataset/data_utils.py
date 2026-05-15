import glob
import os.path as osp
import random
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .WSI import WSIClf


def read_datasplit_npz(path):
    data_npz = np.load(path, allow_pickle=True)
    train = [str(s) for s in data_npz["train_patients"]]
    val = [str(s) for s in data_npz["val_patients"]] if "val_patients" in data_npz else []
    test = [str(s) for s in data_npz["test_patients"]] if "test_patients" in data_npz else []
    return train, val, test


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _norm_join(root, rel_path):
    rel_path = str(rel_path).replace("\\", "/").lstrip("/")
    return osp.join(root, rel_path)


def _auto_find_table_csv(cfg, dataset_name, fold_id):
    root = cfg["dataset_root_dir"]
    cv_cfg = cfg.get("cv", {}) or {}

    if cv_cfg.get("enabled", True):
        cv_subdir = str(cv_cfg.get("subdir", "CV10"))
        cv_csv = osp.join(root, dataset_name, cv_subdir, f"fold_{fold_id:02d}", "all.csv")
        if osp.exists(cv_csv):
            return cv_csv

    if "path_table" in cfg:
        ds_upper = dataset_name.upper()
        table_csv = _norm_join(root, cfg["path_table"].format(ds_upper, ds_upper))
        if osp.exists(table_csv):
            return table_csv

    table_dir = osp.join(root, dataset_name, "table")
    if osp.isdir(table_dir):
        candidates = sorted(glob.glob(osp.join(table_dir, "*.csv")))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"Cannot find table csv for dataset={dataset_name}")


def _auto_find_split_npz(cfg, dataset_name, fold_id):
    root = cfg["dataset_root_dir"]
    if "path_split" in cfg:
        for fold_value in (fold_id, f"{fold_id:02d}"):
            split_path = _norm_join(root, cfg["path_split"].format(dataset_name, fold_value))
            if osp.exists(split_path):
                return split_path

    split_dir = osp.join(root, dataset_name, "datasplit")
    candidates = sorted(glob.glob(osp.join(split_dir, "*.npz")))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cannot find split npz for dataset={dataset_name}")


def _auto_find_feat_dir(cfg, dataset_name):
    root = cfg["dataset_root_dir"]
    arch = cfg.get("base_model_arch", "CONCH")

    if "path_feat" in cfg:
        feat_dir = _norm_join(root, cfg["path_feat"].format(dataset_name, arch))
        if osp.isdir(feat_dir):
            return feat_dir

    feat_dir = osp.join(root, dataset_name, f"feats-l1-s256_{arch}")
    if osp.isdir(osp.join(feat_dir, "pt_files")):
        return osp.join(feat_dir, "pt_files")
    if osp.isdir(feat_dir):
        return feat_dir

    feat_dir = osp.join(root, dataset_name, "pt_files")
    if osp.isdir(feat_dir):
        return feat_dir
    raise FileNotFoundError(f"Cannot find feature directory for dataset={dataset_name}")


def get_data_loaders(cfg):
    print("[info] Loading data...")
    data_loaders = {}
    fold_id = int((cfg.get("cv", {}) or {}).get("fold_id", 1))

    for dataset_name in cfg["dataset_names"]:
        print("*" * 10, dataset_name.upper(), "*" * 10)
        path_table = _auto_find_table_csv(cfg, dataset_name, fold_id)
        path_feat = _auto_find_feat_dir(cfg, dataset_name)
        print(f"[info] Using table: {path_table}")
        print(f"[info] Using feat dir: {path_feat}")

        df = pd.read_csv(path_table, dtype={"patient_id": str})
        if "Set" in df.columns:
            df["Set"] = df["Set"].astype(str).str.strip().str.lower()
            pids_train = df.loc[df["Set"] == "train", "patient_id"].unique().tolist()
            pids_val = df.loc[df["Set"] == "val", "patient_id"].unique().tolist()
            pids_test = df.loc[df["Set"] == "test", "patient_id"].unique().tolist()
        else:
            split_path = _auto_find_split_npz(cfg, dataset_name, fold_id)
            print(f"[info] Using split: {split_path}")
            pids_train, pids_val, pids_test = read_datasplit_npz(split_path)

        print(f"pids_train: count: {len(pids_train)}")
        print(f"pids_val:   count: {len(pids_val)}")
        print(f"pids_test:  count: {len(pids_test)}")

        loaders = {}
        for split, pids in zip(("train", "val", "test"), (pids_train, pids_val, pids_test)):
            dataset = WSIClf(pids, path_feat, path_table, cfg["feat_format"])
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"],
                shuffle=(split == "train"),
                generator=seed_generator(cfg["data_split_seed"]),
                worker_init_fn=seed_worker,
                collate_fn=default_collate,
            )
            loaders[split] = loader
            print(f"sids_{split}: count: {len(dataset)}")

        for split, loader in loaders.items():
            dataset = loader.dataset
            print(f"pids_{split}_label_count:", dict(Counter(dataset.pid2label.values())))
            print(f"sids_{split}_label_count:", dict(Counter(dataset.sid2label.values())))

        data_loaders[dataset_name] = loaders
    return data_loaders


def retrieve_from_table_clf(
    patient_ids,
    table_path,
    ret=None,
    level="slide",
    shuffle=False,
    processing_table=None,
    pid_column="patient_id",
):
    assert level in ["slide", "patient"]
    if ret is None:
        ret = ["pid", "pid2sid", "pid2label"] if level == "patient" else ["sid", "sid2pid", "sid2label"]

    df = pd.read_csv(table_path, dtype={pid_column: str})
    for column in (pid_column, "pathology_id", "label"):
        assert column in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    patient_ids = set(str(pid) for pid in patient_ids)
    df = df[df[pid_column].isin(patient_ids)]

    pid, sid = [], []
    pid2sid, pid2label, sid2pid, sid2label = {}, {}, {}, {}
    for _, row in df.iterrows():
        patient_id = str(row[pid_column])
        slide_id = str(row["pathology_id"])
        label = int(row["label"])

        if patient_id not in pid2sid:
            pid.append(patient_id)
            pid2sid[patient_id] = []
            pid2label[patient_id] = label
        pid2sid[patient_id].append(slide_id)
        sid.append(slide_id)
        sid2pid[slide_id] = patient_id
        sid2label[slide_id] = label

    if shuffle:
        target = pid if level == "patient" else sid
        random.shuffle(target)

    mapping = {
        "pid": pid,
        "sid": sid,
        "pid2sid": pid2sid,
        "sid2pid": sid2pid,
        "pid2label": pid2label,
        "sid2label": sid2label,
    }
    return [mapping[item] for item in ret]


def read_patch_data(path, dtype="torch", key="features"):
    assert dtype in ["numpy", "torch"]
    ext = osp.splitext(path)[1]
    if ext == ".h5":
        with h5py.File(path, "r") as hf:
            data = hf[key][:]
    elif ext == ".pt":
        data = torch.load(path, map_location=torch.device("cpu"))
    elif ext == ".npy":
        data = np.load(path)
    else:
        raise ValueError(f"Not support {ext}")

    if isinstance(data, np.ndarray) and dtype == "torch":
        return torch.from_numpy(data)
    if isinstance(data, Tensor) and dtype == "numpy":
        return data.numpy()
    return data


def get_sids(data_loader):
    return {
        dataset_name: {split: loader.dataset.sids for split, loader in loaders.items()}
        for dataset_name, loaders in data_loader.items()
    }
