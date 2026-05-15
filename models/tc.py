import torch
import torch.nn as nn


class TextualCalibration(nn.Module):
    """TC: learn a small residual for each seen class text prototype."""

    def __init__(self, cfg, tc_residuals):
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg["alpha"]
        self.tc_residuals = tc_residuals

    def forward(self, class_prompt_feature):
        num_tasks = int(self.cfg["task_num"])
        residual = torch.cat([self.tc_residuals[i] for i in range(num_tasks)], dim=0)
        if residual.shape[0] != class_prompt_feature.shape[0]:
            raise RuntimeError(
                "TC residual count does not match class prompt count: "
                f"{residual.shape[0]} vs {class_prompt_feature.shape[0]}"
            )
        return class_prompt_feature + self.alpha * residual


__all__ = ["TextualCalibration"]
