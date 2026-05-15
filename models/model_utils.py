import numpy as np


def freeze_weight(model, model_name):
    for param in model.parameters():
        param.requires_grad = False
    print(f"[setup] weights of {model_name} are frozen.")


def activate_current_tc_residual(tc_residuals, task_num):
    """Train only the current task's TC residual and freeze previous tasks."""
    for param in tc_residuals:
        param.requires_grad = False
        param.grad = None
    tc_residuals[task_num - 1].requires_grad = True


class EarlyStopping:
    def __init__(self, warmup=5, patience=15, verbose=False, threshold=1e-6):
        self.warmup = warmup
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.counter = 0
        self.early_stop = False
        self.save_checkpoint = False
        self.metric_min = np.inf

    def __call__(self, epoch, metric):
        self.save_checkpoint = False
        if epoch <= self.warmup:
            return
        if self.metric_min == np.inf:
            self.update_metric(metric)
            return
        if self.metric_min - metric < self.threshold:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return
        self.counter = 0
        self.update_metric(metric)

    def stop(self, **kws):
        return self.early_stop

    def save_ckpt(self, **kws):
        return self.save_checkpoint

    def update_metric(self, metric):
        if self.verbose:
            print(f"Monitoring metric decreased ({self.metric_min:.6f} --> {metric:.6f}). Saving model.")
        self.metric_min = metric
        self.save_checkpoint = True
