
"""Metric and forgetting helpers for KMP-MIL."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


class MetricsMixin:
    def _compute_metrics(self, y_true: np.ndarray,
                         y_pred_prob: np.ndarray,
                         binary: bool) -> Dict[str, float]:
        if binary:
            prob_pos   = y_pred_prob[:, 1] if y_pred_prob.ndim == 2 else y_pred_prob
            y_pred_lbl = (prob_pos >= 0.5).astype(int)

            acc   = accuracy_score(y_true, y_pred_lbl)
            try:
                auc = roc_auc_score(y_true, prob_pos)
            except ValueError:
                auc = np.nan
            try:
                pr_auc = average_precision_score(y_true, prob_pos)
            except ValueError:
                pr_auc = np.nan

        else:
            y_pred_lbl = np.argmax(y_pred_prob, axis=1)

            acc   = accuracy_score(y_true, y_pred_lbl)
            n_classes = y_pred_prob.shape[1]
            auc_list, ap_list = [], []

            for c in range(n_classes):
                if (y_true == c).sum() == 0:
                    continue
                y_bin = (y_true == c).astype(int)
                try:
                    auc_list.append(roc_auc_score(y_bin, y_pred_prob[:, c]))
                except ValueError:
                    pass
                try:
                    ap_list.append(average_precision_score(y_bin, y_pred_prob[:, c]))
                except ValueError:
                    pass

            auc    = float(np.mean(auc_list)) if len(auc_list) > 0 else np.nan
            pr_auc = float(np.mean(ap_list))  if len(ap_list) > 0 else np.nan

        return {
            'acc': float(acc),
            'auc': float(auc),
            'pr_auc': float(pr_auc),
        }


    def _calc_forgetting(self,
                         task_name: str,
                         cur_metrics: Dict[str, float],
                         base_metrics: Optional[Dict[str, Dict[str, float]]] = None,
                         best_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
        base_store = self.base_metrics if base_metrics is None else base_metrics
        best_store = self.best_metrics if best_metrics is None else best_metrics
        best = best_store.get(task_name, base_store[task_name])

        def _fg(best_v, cur_v):
            if best_v is None or cur_v is None: return np.nan
            if np.isnan(best_v) or np.isnan(cur_v): return np.nan
            return max(0.0, float(best_v) - float(cur_v))

        forget_acc = _fg(best.get('acc', np.nan),  cur_metrics.get('acc',  np.nan))
        forget_auc = _fg(best.get('auc', np.nan),  cur_metrics.get('auc',  np.nan))
        forget_pr  = _fg(best.get('pr_auc', np.nan), cur_metrics.get('pr_auc', np.nan))

        return {
            'forget_acc'   : forget_acc,
            'forget_auc'   : forget_auc,
            'forget_pr_auc': forget_pr,
        }

    def _update_task_cl_forgetting(self,
                                   task_name: str,
                                   cur_metrics: Dict[str, float]) -> Dict[str, float]:
        if task_name not in self.task_cl_base_metrics:
            self.task_cl_base_metrics[task_name] = {
                'acc': cur_metrics.get('acc', np.nan),
                'auc': cur_metrics.get('auc', np.nan),
                'pr_auc': cur_metrics.get('pr_auc', np.nan),
            }

        if task_name not in self.task_cl_best_metrics:
            self.task_cl_best_metrics[task_name] = cur_metrics.copy()
        else:
            bm = self.task_cl_best_metrics[task_name]

            def _max_metric(prev, cur):
                if cur is None or np.isnan(cur):
                    return prev
                if prev is None or np.isnan(prev):
                    return cur
                return max(prev, cur)

            bm['acc'] = _max_metric(bm.get('acc', np.nan), cur_metrics.get('acc', np.nan))
            bm['auc'] = _max_metric(bm.get('auc', np.nan), cur_metrics.get('auc', np.nan))
            bm['pr_auc'] = _max_metric(bm.get('pr_auc', np.nan), cur_metrics.get('pr_auc', np.nan))

        diff = self._calc_forgetting(
            task_name,
            cur_metrics,
            base_metrics=self.task_cl_base_metrics,
            best_metrics=self.task_cl_best_metrics,
        )
        cur_metrics.update(diff)
        return cur_metrics


