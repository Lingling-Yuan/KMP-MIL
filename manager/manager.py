

import os.path as osp
from typing import Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from manager.metrics_helpers import MetricsMixin
from manager.evaluation import EvaluationMixin
from manager.training import TrainingMixin

from utils import (
    get_device, get_loss_function, mkdir_if_missing,
    get_current_class_prompts, get_current_eval_dataloader,
    get_current_task_descriptors,
)

from dataset import get_data_loaders, get_sids
from conch import create_model_from_pretrained
from models import EarlyStopping, freeze_weight, activate_current_tc_residual, KMPMIL
from models.kmp import build_task_kmp_pools, build_universal_kmp_mil, build_task_kmp_mil


class Manager(MetricsMixin, EvaluationMixin, TrainingMixin, object):
    def __init__(self, cfg):



        self.cfg = cfg

        self.device = get_device(cfg['cuda_id'])

        if cfg['base_model_arch'] == 'CONCH':
            self.base_model, _ = create_model_from_pretrained(
                "conch_ViT-B-16",
                checkpoint_path=cfg['conch_ckpt_path'],
                device=self.device
            )
            self.base_model.eval()

            self.base_model.dtype = self.base_model.logit_scale.dtype
            self.embedding_dim = self.base_model.text.ln_final.weight.shape[0]
            self.feature_dim  = self.base_model.visual.proj_contrast.shape[1]
        else:
            raise NotImplementedError("Please specify a valid architecture.")


        freeze_weight(self.base_model, cfg['base_model_arch'])
        self.dtype = self.base_model.dtype

        self.tc_residuals = nn.ParameterList([
            nn.Parameter(
                torch.zeros(int(num_cls), self.feature_dim, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )
            for num_cls in cfg['dataset_subtype_num']
        ])


        self.writer      = None
        self.model       = None
        self.optimizer   = None
        self.lr_scheduler= None
        self.early_stop  = None
        self.universal_model_class_cl = None
        self.final_result = {}


        self.loss_function = get_loss_function(cfg['loss_function'])
        print('[setup] loss function:', cfg['loss_function'])


        self.current_dataset           = None
        self.current_save_result_dir   = None
        self.current_class_prompts     = None


        self.data_loader = get_data_loaders(cfg)
        self.sids        = get_sids(self.data_loader)

        self.base_metrics: Dict[str, Dict[str, float]] = {}
        self.metric_history: List[Dict[str, Dict[str, float]]] = []
        self.best_metrics: Dict[str, Dict[str, float]] = {}
        self.task_cl_base_metrics: Dict[str, Dict[str, float]] = {}
        self.task_cl_best_metrics: Dict[str, Dict[str, float]] = {}


        self.task_cl_metrics_per_task = {}


    @torch.no_grad()
    def _build_task_prototype_pools(self, pool_len: int):
        """
        Create current-task KMP memories and PFC parameters.
        Method-specific construction lives in models/kmp.py; Manager only
        coordinates the continual-training schedule.
        """
        return build_task_kmp_pools(
            self.cfg,
            feature_dim=self.feature_dim,
            embedding_dim=self.embedding_dim,
            dtype=self.dtype,
            device=self.device,
            pool_len=pool_len,
        )


    def _build_universal_model_from_banks(
        self,
        key_bank: List[torch.Tensor],
        prompt_bank: List[torch.Tensor],
        pfc_gamma_delta_bank: Optional[List[torch.Tensor]] = None,
        pfc_beta_bank: Optional[List[torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Build the universal KMP model for class-CL inference.
        """
        return build_universal_kmp_mil(
            cfg=self.cfg,
            base_model=self.base_model,
            device=self.device,
            dtype=self.dtype,
            key_bank=key_bank,
            prompt_bank=prompt_bank,
            tc_residuals=self.tc_residuals,
            current_class_prompts=self.current_class_prompts,
            pfc_gamma_delta_bank=pfc_gamma_delta_bank,
            pfc_beta_bank=pfc_beta_bank,
        )



    def incre_train(self):
        
        num_tasks = len(self.cfg['dataset_names'])


        key_bank: List[torch.Tensor] = []
        prompt_bank: List[torch.Tensor] = []
        pfc_gamma_delta_bank: List[torch.Tensor] = []
        pfc_beta_bank: List[torch.Tensor] = []


        self._kmp_key_bank = key_bank
        self._kmp_prompt_bank = prompt_bank
        self._kmp_pfc_gamma_delta_bank = pfc_gamma_delta_bank
        self._kmp_pfc_beta_bank = pfc_beta_bank

        for task_idx, dataset_name in enumerate(self.cfg['dataset_names']):
            is_last_task = (task_idx == num_tasks - 1)


            self.cfg['task_num'] += 1
            print(f'[train] Task: {self.cfg["task_num"]}, Dataset: {dataset_name}')

            self.current_dataset = dataset_name
            self.current_save_result_dir = osp.join(
                self.cfg['save_result_dir'],
                f'{dataset_name}-task{self.cfg["task_num"]}'
            )
            mkdir_if_missing(self.current_save_result_dir)


            self.current_class_prompts = get_current_class_prompts(
                self.cfg, self.current_dataset
            )


            current_descriptors = get_current_task_descriptors(self.cfg, self.current_dataset)
            if current_descriptors is None:
                current_descriptors = []
            if len(current_descriptors) > 0:
                pool_len = len(current_descriptors)
                print(f"[KMP] Found {pool_len} descriptors for {dataset_name} -> create task-specific pool_len={pool_len}")
            else:
                pool_len = int(self.cfg.get('pool_size', 0))
                print(f"[KMP] Warning: No descriptors for {dataset_name}. Fallback to pool_len=cfg['pool_size']={pool_len}")



            activate_current_tc_residual(self.tc_residuals, self.cfg['task_num'])


            task_key, task_prompt, task_pfc_gamma_delta, task_pfc_beta = self._build_task_prototype_pools(pool_len)


            past_keys_list = [k for k in key_bank]


            self.model = KMPMIL(
                self.cfg, self.base_model, self.device,
                task_key, task_prompt, self.tc_residuals,
                self.current_class_prompts,
                pfc_gamma_delta_pool=task_pfc_gamma_delta,
                pfc_beta_pool=task_pfc_beta,
                past_keys=past_keys_list,
                current_descriptors=current_descriptors
            )

            print('[setup] Model construction completed')

            total_params_model = sum(p.numel() for p in self.model.parameters())
            trainable_params_model = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[Model Params] total(model) = {total_params_model:,} | trainable(model) = {trainable_params_model:,}")


            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.cfg['adam_lr'],
                weight_decay=self.cfg['adam_weight_decay'],
                eps=self.cfg['adam_eps']
            )
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.cfg['lrs_mode'],
                factor=self.cfg['lrs_factor'],
                patience=self.cfg['lrs_patience'][task_idx],
                threshold=self.cfg['lrs_threshold'],
                threshold_mode=self.cfg['lrs_threshold_mode']
            )
            self.early_stop = EarlyStopping(
                warmup=self.cfg['es_warmup'],
                patience=self.cfg['es_patience'][task_idx],
                verbose=self.cfg['es_verbose'],
                threshold=self.cfg['es_threshold']
            )


            self._run_training()


            if self.cfg['load_best_ckpt']:
                print('[infor] loading best checkpoint...')
                best_ckpt_path = osp.join(self.current_save_result_dir, 'model_ckpts', 'best.pth')
                checkpoint = torch.load(best_ckpt_path, map_location=self.device)
                for key, value in checkpoint.items():
                    if ('key' in key) or ('prompt' in key) or ('tc_residuals' in key) or ('pfc' in key):
                        try:
                            self.model.state_dict()[key].copy_(value)
                        except Exception as e:
                            print(f"[Warn] Loading {key} failed: {e}")


            with torch.no_grad():
                cur_keys = torch.cat([p.detach().cpu() for p in self.model.key], dim=0).clone()
                cur_prompts = torch.cat([p.detach().cpu() for p in self.model.prompt], dim=0).clone()
                key_bank.append(cur_keys)
                prompt_bank.append(cur_prompts)

                pfc_module = getattr(self.model, 'pfc', None)
                if getattr(self.model, 'use_pfc', False) and (pfc_module is not None):
                    cur_gamma = torch.cat([p.detach().cpu() for p in pfc_module.gamma_delta_pool], dim=0).clone()
                    cur_beta  = torch.cat([p.detach().cpu() for p in pfc_module.beta_pool], dim=0).clone()
                    pfc_gamma_delta_bank.append(cur_gamma)
                    pfc_beta_bank.append(cur_beta)
                else:
                    cur_gamma = torch.zeros_like(cur_keys)
                    cur_beta = torch.zeros_like(cur_keys)
                    pfc_gamma_delta_bank.append(cur_gamma)
                    pfc_beta_bank.append(cur_beta)

                print(f"[KMP] Task {task_idx+1} pool cached. "
                      f"bank_tasks={len(key_bank)}, cur_pool_len={cur_keys.shape[0]}, total_pool_len={sum(k.shape[0] for k in key_bank)}")


            self.universal_model_class_cl = self._build_universal_model_from_banks(
                key_bank=key_bank,
                prompt_bank=prompt_bank,
                pfc_gamma_delta_bank=pfc_gamma_delta_bank if bool(self.cfg.get('use_pfc', True)) else None,
                pfc_beta_bank=pfc_beta_bank if bool(self.cfg.get('use_pfc', True)) else None,
            )


            eval_result_dir = osp.join(self.current_save_result_dir, 'eval_results')
            mkdir_if_missing(eval_result_dir)
            cur_eval_loader = get_current_eval_dataloader(
                self.data_loader, self.current_dataset
            )


            self._eval_class_cl_all(
                cur_eval_loader,
                eval_result_dir,
                print_mean=is_last_task,
                model_override=self.universal_model_class_cl
            )
            print('[eval at val/test dataset] result dir:', eval_result_dir)



            cur_task_eval_loader = {
                k: v for k, v in cur_eval_loader.items()
                if k.startswith(f"{self.current_dataset}/")
            }

            task_cl_metrics = self._eval_task_cl_metrics(
                cur_task_eval_loader,
                eval_result_dir,
                model_override=self.model,
                use_universal_by_default=False
            )
            self.task_cl_metrics_per_task[self.current_dataset] = task_cl_metrics or {}
            print('[cal task-CL metrics] result dir:', eval_result_dir)




            if is_last_task:
                final_task_cl_dir = osp.join(eval_result_dir, "final_task_cl_all_tasks")
                mkdir_if_missing(final_task_cl_dir)


                final_all_task_cl = {}


                full_class_prompts = self.current_class_prompts


                orig_ds = self.current_dataset
                for i, ds in enumerate(self.cfg['dataset_names']):

                    tmp_model = self._build_task_model_from_pool(
                        task_keys=key_bank[i],
                        task_prompts=prompt_bank[i],
                        pfc_gamma_delta=pfc_gamma_delta_bank[i] if i < len(pfc_gamma_delta_bank) else None,
                        pfc_beta=pfc_beta_bank[i] if i < len(pfc_beta_bank) else None,
                        class_prompts=full_class_prompts
                    )


                    ds_eval_loader = {k: v for k, v in cur_eval_loader.items() if k.startswith(f"{ds}/")}


                    self.current_dataset = ds
                    m_ds = self._eval_task_cl_metrics(
                        ds_eval_loader,
                        final_task_cl_dir,
                        model_override=tmp_model,
                        use_universal_by_default=False
                    )
                    final_all_task_cl[ds] = m_ds or {}


                    for sp in ('val', 'test'):
                        if sp in (m_ds or {}):
                            md = m_ds[sp]
                            acc = md.get('acc', float('nan'))
                            auc = md.get('auc', float('nan'))
                            pr  = md.get('pr_auc', float('nan'))
                            f_acc = md.get('forget_acc', float('nan'))
                            f_auc = md.get('forget_auc', float('nan'))
                            f_pr = md.get('forget_pr_auc', float('nan'))
                            print(f"[FINAL][task-CL][{ds}][{sp}] "
                                  f"acc={acc:.6f} auc={auc:.6f} pr_auc={pr:.6f} "
                                  f"forget_acc={f_acc:.6f} forget_auc={f_auc:.6f} forget_pr_auc={f_pr:.6f}")


                self.current_dataset = orig_ds


                self.task_cl_metrics_per_task_final = final_all_task_cl
                self.task_cl_metrics_per_task = final_all_task_cl





        if self.metric_history:
            final_snapshot = self.metric_history[-1]

            def mean_key(k):
                arr = np.array([m.get(k, np.nan) for m in final_snapshot.values()], dtype=float)
                arr = arr[~np.isnan(arr)]
                return float(np.nan) if arr.size == 0 else float(np.mean(arr))

            avg_acc         = mean_key('acc')
            avg_auc         = mean_key('auc')
            avg_pr_auc      = mean_key('pr_auc')
            avg_fgt_acc     = mean_key('forget_acc')
            avg_fgt_auc     = mean_key('forget_auc')
            avg_fgt_pr      = mean_key('forget_pr_auc')

            print('[result] Final-round Averages -> '
                  f'ACC={avg_acc:.6f}, '
                  f'AUC={avg_auc:.6f}, PR-AUC={avg_pr_auc:.6f}, '
                  f'Forget_ACC={avg_fgt_acc:.6f}, '
                  f'Forget_AUC={avg_fgt_auc:.6f}, '
                  f'Forget_PR-AUC={avg_fgt_pr:.6f}')

            self.final_result = {
                'acc': float(avg_acc),
                'auc': float(avg_auc),
                'pr_auc': float(avg_pr_auc),
                'forget_acc': float(avg_fgt_acc),
                'forget_auc': float(avg_fgt_auc),
                'forget_pr_auc': float(avg_fgt_pr),
            }


        return getattr(self, 'final_result', {})

    def _build_task_model_from_pool(
        self,
        task_keys: torch.Tensor,
        task_prompts: torch.Tensor,
        pfc_gamma_delta: Optional[torch.Tensor] = None,
        pfc_beta: Optional[torch.Tensor] = None,
        class_prompts: Optional[dict] = None,
    ) -> nn.Module:
        
        if class_prompts is None:
            class_prompts = self.current_class_prompts

        return build_task_kmp_mil(
            cfg=self.cfg,
            base_model=self.base_model,
            device=self.device,
            dtype=self.dtype,
            task_keys=task_keys,
            task_prompts=task_prompts,
            tc_residuals=self.tc_residuals,
            class_prompts=class_prompts,
            pfc_gamma_delta=pfc_gamma_delta,
            pfc_beta=pfc_beta,
        )

    def _save_model(self, epoch, ckpt_type='best'):
        net_ckpt_dict = self._get_state_dict(epoch)
        save_dir = osp.join(self.current_save_result_dir, 'model_ckpts')
        mkdir_if_missing(save_dir)
        torch.save(net_ckpt_dict, save_dir + f'/{ckpt_type}.pth')

    def _get_state_dict(self, epoch=None):
        return_dict = {'epoch': epoch}

        for k, v in self.model.state_dict().items():
            if ('key' in k) or ('prompt' in k) or ('tc_residuals' in k) or ('pfc' in k):
                return_dict[k] = v
        return return_dict

