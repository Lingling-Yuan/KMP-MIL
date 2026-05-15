
"""Evaluation helpers for class-CL and task-CL inference."""

import json
import os.path as osp
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.functional import softmax

from utils import mkdir_if_missing


class EvaluationMixin:
    @torch.no_grad()
    def eval_model(self, model, loader, ckpt_path=None):
        if ckpt_path is not None:
            pass
        model.eval()

        idx_collector, x_collector, y_collector = [], [], []
        all_idx, all_pred, all_gt = [], [], []
        eval_every_batch = int(self.cfg["eval_every_batch"])

        def flush_batch():
            cur_pred, _ = model(x_collector, eval=True)
            all_pred.append(cur_pred.detach().cpu())
            all_gt.append(torch.cat(y_collector, dim=0))
            all_idx.append(torch.cat(idx_collector, dim=0))

        for i_batch, (data_idx, data_x, data_y) in enumerate(loader, start=1):
            x_collector.append(data_x.to(self.device))
            y_collector.append(data_y)
            idx_collector.append(data_idx)
            if i_batch % eval_every_batch == 0:
                flush_batch()
                idx_collector, x_collector, y_collector = [], [], []
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.cfg["cuda_id"])
                    torch.cuda.empty_cache()

        if len(x_collector) > 0:
            flush_batch()

        return {
            "pred": {
                "y": torch.cat(all_gt, dim=0).squeeze(1),
                "y_hat": torch.cat(all_pred, dim=0),
                "idx": torch.cat(all_idx, dim=0).squeeze(1),
            }
        }

    def _eval_and_print_task_cl(self, name: str,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               binary: bool):
        

        metrics = self._compute_metrics(y_true, y_pred, binary)





        if name not in self.base_metrics:
            self.base_metrics[name] = metrics

        return metrics

    def _eval_and_print_class_cl(self, name: str,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 binary: bool):
        ds_name = name.split('/')[0]
        metrics = self._compute_metrics(y_true, y_pred, binary)
        if ds_name in self.base_metrics:
            diff = self._calc_forgetting(ds_name, metrics)
            metrics.update(diff)
        return metrics

    def _eval_task_cl_metrics(
        self,
        data_loaders,
        task_cl_eval_result_dir,
        model_override = None,
        use_universal_by_default: bool = True,
    ):
        
        import os.path as osp
        import pandas as pd


        model_to_use = model_override
        if model_to_use is None and use_universal_by_default:
            model_to_use = getattr(self, 'universal_model_class_cl', None)
        if model_to_use is None:
            model_to_use = self.model


        split_metrics_all: Dict[str, Dict[str, Dict[str, float]]] = {'val': {}, 'test': {}}

        for k, loader in data_loaders.items():

            cltor = self.eval_model(model_to_use, loader)['pred']

            parts = k.split('/')
            ds_name = parts[0]
            split_name = parts[1] if len(parts) > 1 else 'all'

            if ds_name not in self.cfg['dataset_names']:
                continue
            idx_ds = self.cfg['dataset_names'].index(ds_name)


            shift = int(self.cfg['dataset_label_shift'][idx_ds])
            num_sub = int(self.cfg['dataset_subtype_num'][idx_ds])

            y_true = cltor['y'].numpy()


            y_pred = softmax(cltor['y_hat'][:, shift:shift + num_sub], dim=1).numpy()
            binary = (num_sub == 2)


            self._eval_and_print_task_cl(f"{ds_name}/{split_name}", y_true, y_pred, binary)


            metrics = self._compute_metrics(y_true, y_pred, binary)
            if split_name == 'test':
                metrics = self._update_task_cl_forgetting(ds_name, metrics)
                print(f"[result][task-CL][{ds_name}] "
                      f"ACC={metrics['acc']:.6f}, AUC={metrics['auc']:.6f}, PR-AUC={metrics['pr_auc']:.6f}, "
                      f"Forget_ACC={metrics['forget_acc']:.6f}, "
                      f"Forget_AUC={metrics['forget_auc']:.6f}, "
                      f"Forget_PR-AUC={metrics['forget_pr_auc']:.6f}")


            sids = self._get_unique_sids(k, cltor['idx'])
            save_path = f"{task_cl_eval_result_dir}/task-cl-{k.replace('/', '-')}.csv"
            self._save_prediction_clf(sids, y_true, y_pred, save_path, binary=binary)


            try:
                pd.DataFrame([metrics]).to_csv(
                    osp.join(task_cl_eval_result_dir, f"task-cl-metrics-{k.replace('/', '-')}.csv"),
                    index=False, encoding='utf-8'
                )
            except Exception:
                pass


            if split_name in ('val', 'test'):
                split_metrics_all[split_name][ds_name] = metrics


        self.last_task_cl_metrics_all = split_metrics_all


        cur_ds = getattr(self, 'current_dataset', None)
        if cur_ds is None:
            cur_ds = self.cfg['dataset_names'][self.cfg['task_num'] - 1]

        split_metrics_task: Dict[str, Dict[str, float]] = {}
        for sp in ('val', 'test'):
            if (sp in split_metrics_all) and (cur_ds in split_metrics_all[sp]):
                split_metrics_task[sp] = split_metrics_all[sp][cur_ds]

        return split_metrics_task


    def _eval_class_cl_all(self,
                  data_loaders: Dict[str, torch.utils.data.DataLoader],
                  eval_result_dir: str,
                  print_mean: bool = False,
                  model_override = None):
        
        model_to_use = model_override if model_override is not None else self.model

        snapshot: Dict[str, Dict[str, float]] = {}

        for k, loader in data_loaders.items():

            cltor = self.eval_model(model_to_use, loader)['pred']
            ds_name, split = k.split('/')[0], k.split('/')[-1]


            idx_ds = self.cfg['dataset_names'].index(ds_name)
            shift  = int(self.cfg['dataset_label_shift'][idx_ds])
            num_sub = int(self.cfg['dataset_subtype_num'][idx_ds])

            y_true = (cltor['y'] + shift).numpy()


            y_pred_logits = cltor['y_hat']
            y_pred_probs  = torch.softmax(y_pred_logits, dim=1).cpu().numpy()
            binary_this   = (y_pred_probs.shape[1] == 2)


            metrics = self._eval_and_print_class_cl(
                f'{ds_name}/{split}', y_true, y_pred_probs, binary_this)


            if print_mean and split == 'test':
                mu = y_pred_logits.mean(dim=0).cpu()
                msg_mu = ', '.join([f'{i}:{m:.2f}' for i, m in enumerate(mu)])
                print(f'[logit mean] {ds_name}/test -> {msg_mu}')


            sids = self._get_unique_sids(k, cltor['idx'])
            save_path = f"{eval_result_dir}/{k.replace('/', '-')}.csv"
            self._save_prediction_clf(sids, y_true, y_pred_probs, save_path,
                                      binary=binary_this)


            if split == 'test':
                if ds_name not in self.base_metrics:
                    self.base_metrics[ds_name] = {
                        'acc': metrics.get('acc', np.nan),
                        'auc': metrics.get('auc', np.nan),
                        'pr_auc': metrics.get('pr_auc', np.nan),
                    }


                if ds_name not in self.best_metrics:
                    self.best_metrics[ds_name] = metrics.copy()
                else:
                    bm = self.best_metrics[ds_name]

                    def _max_metric(prev, cur):
                        if cur is None or np.isnan(cur):
                            return prev
                        if prev is None or np.isnan(prev):
                            return cur
                        return max(prev, cur)

                    bm['acc'] = _max_metric(bm.get('acc', np.nan), metrics.get('acc', np.nan))
                    bm['auc'] = _max_metric(bm.get('auc', np.nan), metrics.get('auc', np.nan))
                    bm['pr_auc'] = _max_metric(bm.get('pr_auc', np.nan), metrics.get('pr_auc', np.nan))

                diff = self._calc_forgetting(ds_name, metrics)
                metrics.update(diff)
                print(f"[result][class-CL][{ds_name}] "
                      f"ACC={metrics['acc']:.6f}, AUC={metrics['auc']:.6f}, PR-AUC={metrics['pr_auc']:.6f}, "
                      f"Forget_ACC={metrics['forget_acc']:.6f}, "
                      f"Forget_AUC={metrics['forget_auc']:.6f}, "
                      f"Forget_PR-AUC={metrics['forget_pr_auc']:.6f}")

                snapshot[ds_name] = metrics

        if snapshot:
            self.metric_history.append(snapshot)


    def _get_unique_sids(self, k, idxs, concat=None):
        
        dataset_name  = k.split('/')[0]
        dataset_split = k.split('/')[-1]
        sids = self.sids[dataset_name][dataset_split]

        idxs = idxs.tolist()
        if concat is None:

            return [sids[i] for i in idxs]
        else:

            return [sids[v] + "-" + str(concat[i].item()) for i, v in enumerate(idxs)]

    def _save_prediction_clf(self,
                             sids,
                             y_true,
                             y_pred,
                             save_path,
                             binary: bool = True,
                             forgetting: bool = False):
        

        if isinstance(y_true, Tensor):
            y_true = y_true.numpy()

        if isinstance(y_pred, Tensor):

            y_pred = softmax(y_pred, dim=1).numpy()


        assert len(sids) == len(y_true), "sids and y_true length mismatch"
        assert len(sids) == len(y_pred), "sids and y_pred length mismatch"


        save_data = {'sids': sids, 'y': y_true}
        cols = ['sids', 'y']

        if binary:

            save_data['y_hat'] = y_pred[:, 1]
            cols.append('y_hat')
        else:

            for i in range(y_pred.shape[-1]):
                col_name = f'y_hat_{i}'
                save_data[col_name] = y_pred[:, i]
                cols.append(col_name)


        df = pd.DataFrame(save_data, columns=cols)
        df.to_csv(save_path, index=False)

    @torch.no_grad()
    def _eval_and_print(self, cltor, name='', at_epoch=None, if_binary=True):
        

        y_true_t = cltor['y']
        logits_t = cltor['y_hat']


        num_cls = logits_t.shape[1]
        y_min = int(torch.min(y_true_t).item())
        y_max = int(torch.max(y_true_t).item())
        if y_max >= num_cls or y_min < 0:
            uniq = torch.unique(y_true_t).sort().values
            if len(uniq) == num_cls and (int(uniq[-1].item()) - int(uniq[0].item()) + 1 == num_cls):
                y_true_local = (y_true_t - int(uniq[0].item())).to(dtype=torch.long, device=logits_t.device)
            else:
                mapping = {int(l.item()): i for i, l in enumerate(uniq)}
                y_true_local = torch.tensor(
                    [mapping[int(l.item())] for l in y_true_t.cpu()],
                    dtype=torch.long, device=logits_t.device
                )
        else:
            y_true_local = y_true_t.to(dtype=torch.long, device=logits_t.device)


        with torch.no_grad():
            loss_val = self.loss_function(logits_t, y_true_local).item()


        y_true = y_true_local.cpu().numpy()
        y_pred = torch.softmax(logits_t, dim=1).cpu().numpy()
        metrics = self._compute_metrics(y_true, y_pred, if_binary)


        metrics['loss'] = loss_val


        msg = f'[{name}]'
        if at_epoch is not None:
            msg += f' At epoch {at_epoch}:'
        msg += (f' ACC={metrics.get("acc", 0.0):.6f}, AUC={metrics.get("auc", 0.0):.6f}, '
                f'PR-AUC={metrics.get("pr_auc", 0.0):.6f}, loss={metrics["loss"]:.6f}')
        print(msg)

        return metrics

