import torch
import torch.nn as nn


class PrototypeConditionedFeatureCalibration(nn.Module):
    """PFC: use matched prototypes to modulate patch features before MIL aggregation."""

    def __init__(self, cfg, device, dtype, gamma_delta_pool, beta_pool):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.gamma_delta_pool = gamma_delta_pool
        self.beta_pool = beta_pool
        self.pfc_logit_scale = float(cfg.get("pfc_logit_scale", 10.0))
        self.gamma_scale = float(cfg.get("pfc_gamma_scale", 0.5))
        self.beta_scale = float(cfg.get("pfc_beta_scale", 0.2))
        self.reg_type = str(cfg.get("pfc_reg_type", "l2")).lower()

    def _merge_pool(self, pool_plist):
        return torch.cat([p for p in pool_plist], dim=0)

    def forward(self, x_list, indices, proto_scores):
        weights = torch.softmax(proto_scores * self.pfc_logit_scale, dim=1).to(self.dtype)

        gamma_delta_pool = self._merge_pool(self.gamma_delta_pool).to(self.device).type(self.dtype)
        beta_pool = self._merge_pool(self.beta_pool).to(self.device).type(self.dtype)

        gamma_delta_sel = gamma_delta_pool[indices]
        beta_sel = beta_pool[indices]
        weights = weights.unsqueeze(-1)

        gamma_delta_hat = (weights * gamma_delta_sel).sum(dim=1)
        beta_hat_raw = (weights * beta_sel).sum(dim=1)
        gamma_hat = 1.0 + self.gamma_scale * torch.tanh(gamma_delta_hat)
        beta_hat = self.beta_scale * torch.tanh(beta_hat_raw)

        x_list_mod = []
        for i, x in enumerate(x_list):
            gamma_i = gamma_hat[i].view(1, 1, -1)
            beta_i = beta_hat[i].view(1, 1, -1)
            x_list_mod.append(x.type(self.dtype) * gamma_i + beta_i)

        if self.reg_type == "l1":
            reg_loss = gamma_delta_hat.abs().mean() + beta_hat_raw.abs().mean()
        else:
            reg_loss = gamma_delta_hat.pow(2).mean() + beta_hat_raw.pow(2).mean()

        debug = {
            "pfc_w": weights.squeeze(-1).float(),
            "pfc_gamma_hat": gamma_hat.float(),
            "pfc_beta_hat": beta_hat.float(),
        }
        return x_list_mod, reg_loss.float(), debug


__all__ = ["PrototypeConditionedFeatureCalibration"]
