import argparse
import errno
import json
import os
import os.path as osp
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.manifold import TSNE
from torch.nn import functional as F

from .logger import setup_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-f", required=True, type=str, help="path to the config file")
    parser.add_argument("--seed", "-s", type=int, default=1, help="random number seed and data fold")
    parser.add_argument("--time", "-t", type=str, default=None, help="run timestamp")
    args = vars(parser.parse_args())
    if args["time"] is None:
        args["time"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return args


def get_config(config_path):
    with open(config_path, "r", encoding="utf-8") as setting:
        return yaml.load(setting, Loader=yaml.FullLoader)


def mkdir_if_missing(dirname):
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def create_result_dir(result_dir, cfg):
    config_name = osp.splitext(osp.basename(cfg["config"]))[0]
    time_name = str(cfg["time"]).replace(":", "-")
    result_dir = osp.join(result_dir, config_name, time_name, f"train-data_split_seed_{cfg['seed']}")
    mkdir_if_missing(result_dir)
    return result_dir


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[setup] seed: {seed}")


def set_config(config, cfg, save_result_dir):
    config_dir = osp.dirname(osp.abspath(cfg["config"]))
    for key in ("class_prompts_path", "task_descriptors_path"):
        if key in config and not osp.isabs(config[key]):
            config[key] = osp.join(config_dir, config[key])

    config["config_file"] = cfg["config"]
    config["seed"] = cfg["seed"]
    config["time"] = cfg["time"]
    config["data_split_seed"] = cfg["seed"]
    config["save_result_dir"] = save_result_dir


def init(cfg, config):
    save_result_dir = create_result_dir(config["result_dir"], cfg)
    setup_logger(save_result_dir)
    set_random_seed(cfg["seed"])
    set_config(config, cfg, save_result_dir)


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key, val in config.items():
        print(f"{key:<24} -->   {val}")
    print("**************** MODEL CONFIGURATION ****************")


def get_device(gpu_id):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def get_loss_function(loss_function_name):
    if loss_function_name == "cross_entropy":
        return F.cross_entropy
    raise NotImplementedError("Please specify a valid loss function.")


def get_init_key_frequency(config):
    return {dataset_name: [] for dataset_name in config["dataset_names"]}


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_current_class_prompts(config, current_dataset):
    data = _read_json(config["class_prompts_path"])
    classnames = data["classnames"]
    templates = data["templates"]

    current = {"class_prompts": [], "count": []}
    for dataset_name, dataset_classes in classnames.items():
        for names in dataset_classes.values():
            current["count"].append(len(names) * len(templates))
            for name in names:
                current["class_prompts"].extend(
                    template.replace("CLASSNAME", name) for template in templates
                )
        if dataset_name == current_dataset:
            break
    return current


def get_current_task_descriptors(config, current_dataset):
    descriptors = _read_json(config["task_descriptors_path"])
    return list(descriptors.get(current_dataset, []))


def check_feature_distribution(feature, label, save_dir, type="image", epoch=0):
    if type == "text":
        label = torch.arange(feature.shape[1]).unsqueeze(0).repeat(feature.shape[0], 1).view(-1)
        feature = feature.view(-1, feature.shape[-1])

    print(f"[epoch {epoch}] check {type} feature distribution")
    if torch.isnan(feature).any():
        print("[error] Contains NaN")
    if torch.isinf(feature).any():
        print("[error] Contains Inf")

    tsne = TSNE()
    embedded = tsne.fit_transform(feature.numpy())

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    num_cluster = int(torch.max(label).item() + 1)
    sns.set(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    palette = sns.color_palette("bright", num_cluster)
    label_color_mapping = {i: palette[i] for i in range(num_cluster)}
    sns.scatterplot(
        x=embedded[:, 0],
        y=embedded[:, 1],
        hue=label.numpy(),
        legend=False,
        palette=label_color_mapping,
        ax=ax,
        s=3,
        edgecolors="none",
        linewidths=0,
    )
    fig.savefig(osp.join(save_dir, f"{type}_{epoch}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_key_matching_heatmap(value, path, dataset="train"):
    value = torch.stack(value, dim=0).numpy()
    plt.figure(dpi=500)
    plt.imshow(value, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.savefig(osp.join(path, f"{dataset}_heatmap.png"))
    plt.close()


def get_current_eval_dataloader(data_loader_dict, current_dataset):
    current_eval_dataloader = {}
    for key, val in data_loader_dict.items():
        current_eval_dataloader[f"{key}/val"] = val["val"]
        current_eval_dataloader[f"{key}/test"] = val["test"]
        if key == current_dataset:
            break
    return current_eval_dataloader
