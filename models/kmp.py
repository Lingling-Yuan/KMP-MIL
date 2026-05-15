"""Knowledge Memory Pool (KMP) construction utilities for KMP-MIL."""

from typing import List, Optional

import torch
import torch.nn as nn

from .model_il import KMPMIL


def build_task_kmp_pools(cfg, feature_dim: int, embedding_dim: int, dtype, device, pool_len: int):
    """Create the task-specific KMP key/prompt pools and optional PFC pools."""
    task_key = nn.ParameterList([
        nn.Parameter(
            0.02 * torch.randn(1, feature_dim, dtype=dtype, device=device),
            requires_grad=True,
        )
        for _ in range(pool_len)
    ])
    task_prompt = nn.ParameterList([
        nn.Parameter(
            0.02 * torch.randn(1, cfg["prompt_length"], embedding_dim, dtype=dtype, device=device),
            requires_grad=True,
        )
        for _ in range(pool_len)
    ])

    if bool(cfg.get("use_pfc", True)):
        task_pfc_gamma_delta = nn.ParameterList([
            nn.Parameter(torch.zeros(1, feature_dim, dtype=dtype, device=device), requires_grad=True)
            for _ in range(pool_len)
        ])
        task_pfc_beta = nn.ParameterList([
            nn.Parameter(torch.zeros(1, feature_dim, dtype=dtype, device=device), requires_grad=True)
            for _ in range(pool_len)
        ])
    else:
        task_pfc_gamma_delta = None
        task_pfc_beta = None

    return task_key, task_prompt, task_pfc_gamma_delta, task_pfc_beta


def _rows_to_parameter_list(tensor: torch.Tensor, device, dtype) -> nn.ParameterList:
    return nn.ParameterList([
        nn.Parameter(tensor[i:i + 1].to(device).type(dtype), requires_grad=False)
        for i in range(int(tensor.shape[0]))
    ])


def build_universal_kmp_mil(
    cfg,
    base_model,
    device,
    dtype,
    key_bank: List[torch.Tensor],
    prompt_bank: List[torch.Tensor],
    tc_residuals: nn.ParameterList,
    current_class_prompts,
    pfc_gamma_delta_bank: Optional[List[torch.Tensor]] = None,
    pfc_beta_bank: Optional[List[torch.Tensor]] = None,
) -> nn.Module:
    """Build a class-CL model by concatenating all seen task KMP memories."""
    assert len(key_bank) > 0, "key_bank is empty; cannot build universal KMP model"
    assert len(prompt_bank) == len(key_bank), "prompt_bank and key_bank length mismatch"

    keys_cat = torch.cat([k.detach().cpu() for k in key_bank], dim=0)
    prompts_cat = torch.cat([p.detach().cpu() for p in prompt_bank], dim=0)
    key_plist = _rows_to_parameter_list(keys_cat, device, dtype)
    prompt_plist = _rows_to_parameter_list(prompts_cat, device, dtype)

    use_pfc = bool(cfg.get("use_pfc", True)) and pfc_gamma_delta_bank is not None and pfc_beta_bank is not None
    if use_pfc:
        gamma_cat = torch.cat([g.detach().cpu() for g in pfc_gamma_delta_bank], dim=0)
        beta_cat = torch.cat([b.detach().cpu() for b in pfc_beta_bank], dim=0)
        assert gamma_cat.shape[0] == keys_cat.shape[0] and beta_cat.shape[0] == keys_cat.shape[0], "PFC bank length mismatch"
        gamma_plist = _rows_to_parameter_list(gamma_cat, device, dtype)
        beta_plist = _rows_to_parameter_list(beta_cat, device, dtype)
    else:
        gamma_plist = None
        beta_plist = None

    model = KMPMIL(
        cfg, base_model, device,
        key_plist, prompt_plist, tc_residuals,
        current_class_prompts,
        pfc_gamma_delta_pool=gamma_plist,
        pfc_beta_pool=beta_plist,
        past_keys=None,
        current_descriptors=None,
    )
    model.eval()
    return model


def build_task_kmp_mil(
    cfg,
    base_model,
    device,
    dtype,
    task_keys: torch.Tensor,
    task_prompts: torch.Tensor,
    tc_residuals: nn.ParameterList,
    class_prompts,
    pfc_gamma_delta: Optional[torch.Tensor] = None,
    pfc_beta: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build a task-CL model using only one task-specific KMP segment."""
    assert task_keys.dim() == 2, f"task_keys must be [M,C], got {task_keys.shape}"
    assert task_prompts.dim() == 3, f"task_prompts must be [M,L,D], got {task_prompts.shape}"
    assert task_prompts.shape[0] == task_keys.shape[0], "task_prompts length != task_keys length"

    key_plist = _rows_to_parameter_list(task_keys, device, dtype)
    prompt_plist = _rows_to_parameter_list(task_prompts, device, dtype)

    use_pfc = bool(cfg.get("use_pfc", True)) and pfc_gamma_delta is not None and pfc_beta is not None
    if use_pfc:
        assert pfc_gamma_delta.shape[0] == task_keys.shape[0] and pfc_beta.shape[0] == task_keys.shape[0], "PFC length mismatch"
        gamma_plist = _rows_to_parameter_list(pfc_gamma_delta, device, dtype)
        beta_plist = _rows_to_parameter_list(pfc_beta, device, dtype)
    else:
        gamma_plist = None
        beta_plist = None

    model = KMPMIL(
        cfg, base_model, device,
        key_plist, prompt_plist, tc_residuals,
        class_prompts,
        pfc_gamma_delta_pool=gamma_plist,
        pfc_beta_pool=beta_plist,
        past_keys=None,
        current_descriptors=None,
    )
    model.eval()
    return model


__all__ = [
    "KMPMIL",
    "build_task_kmp_pools",
    "build_universal_kmp_mil",
    "build_task_kmp_mil",
]
