
"""Training loop helpers for KMP-MIL."""

import time

import numpy as np
import torch


class TrainingMixin:
    def _run_training(self):
        

        t_train0 = time.time()
        self.last_train_sec = None


        last_epoch = -1

        for epoch in range(self.cfg['epochs'][self.cfg['task_num'] - 1]):
            last_epoch = epoch + 1
            fold_id = int((self.cfg.get('cv', {}) or {}).get('fold_id', self.cfg.get('data_split_seed', 0)))
            print('[train] fold: {}, task: {}, epoch: {}'.format(
                  fold_id, self.cfg['task_num'], last_epoch))
            print('[train] lr={:.8f}'.format(self.optimizer.param_groups[0]['lr']))

            train_cltor = self._train_each_epoch()
            for k_cltor, v_cltor in train_cltor.items():
                shift   = self.cfg['dataset_label_shift'][self.cfg['task_num'] - 1]
                num_sub = self.cfg['dataset_subtype_num'][self.cfg['task_num'] - 1]
                v_cltor['y_hat'] = v_cltor['y_hat'][:, shift:shift+num_sub]
                if_binary = (num_sub == 2)
                self._eval_and_print(
                    v_cltor, name='train/' + k_cltor,
                    at_epoch=last_epoch, if_binary=if_binary)


            eval_cltor = self.eval_model(self.model, self.data_loader[self.current_dataset]['val'])
            for k_cltor, v_cltor in eval_cltor.items():
                v_cltor['y'] += self.cfg['dataset_label_shift'][self.cfg['task_num'] - 1]

                shift   = self.cfg['dataset_label_shift'][self.cfg['task_num'] - 1]
                num_sub = self.cfg['dataset_subtype_num'][self.cfg['task_num'] - 1]
                v_cltor['y_hat'] = v_cltor['y_hat'][:, shift:shift+num_sub]
                if_binary = (num_sub == 2)

                eval_results = self._eval_and_print(
                    v_cltor, name='val/' + k_cltor,
                    at_epoch=last_epoch, if_binary=if_binary)


                val_loss = eval_results['loss']
                val_acc  = eval_results['acc']
                monitor_metric = val_loss if self.cfg['only_val_loss'] \
                                 else (val_loss + (1 - val_acc)) / 2


            if last_epoch > self.cfg['lrs_warmup']:
                self.lr_scheduler.step(monitor_metric)


            self.early_stop(last_epoch, monitor_metric)
            if self.early_stop.save_ckpt():
                self._save_model(last_epoch, ckpt_type='best')
                print("[save best model] best model saved at epoch {}".format(last_epoch))

            if self.early_stop.stop():
                break


        self._save_model(last_epoch, ckpt_type='last')
        print("[save last model] last model saved at epoch {}".format(last_epoch))

        self.last_train_sec = float(time.time() - t_train0)


    def _train_each_epoch(self):
        self.model.eval()

        x_collector, y_collector = [], []
        all_pred, all_gt = [], []

        i_batch = 0
        train_loader = self.data_loader[self.current_dataset]['train']
        bp_every_batch = self.cfg['bp_every_batch']

        for _, data_x, data_y in train_loader:
            i_batch += 1


            data_y += self.cfg['dataset_label_shift'][self.cfg['task_num'] - 1]
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)

            x_collector.append(data_x)
            y_collector.append(data_y)

            if i_batch % bp_every_batch == 0:

                cur_pred, _ = self._update_network(
                    i_batch, x_collector, y_collector
                )
                all_pred.append(cur_pred.detach().cpu())
                all_gt.append(torch.cat(y_collector, dim=0).detach().cpu())

                x_collector, y_collector = [], []
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.cfg['cuda_id'])
                    torch.cuda.empty_cache()


        if len(x_collector) > 0:
            cur_pred, _ = self._update_network(
                i_batch, x_collector, y_collector
            )
            all_pred.append(cur_pred.detach().cpu())
            all_gt.append(torch.cat(y_collector, dim=0).detach().cpu())

            x_collector, y_collector = [], []
            if torch.cuda.is_available():
                torch.cuda.set_device(self.cfg['cuda_id'])
                torch.cuda.empty_cache()

        all_pred = torch.cat(all_pred, dim=0)
        all_gt   = torch.cat(all_gt,   dim=0).squeeze(1)

        train_cltor = dict()
        train_cltor['pred'] = {
            'y': all_gt,
            'y_hat': all_pred,
        }
        return train_cltor



    def _update_network(self, i_batch, xs, ys):
        
        bag_preds, loss_dict, key_indices = self.model(xs)

        self.optimizer.zero_grad()

        bag_label = torch.cat(ys, dim=0).squeeze(-1)
        clf_loss = self.loss_function(bag_preds, bag_label)

        pfc_w = float(self.cfg.get('pfc_reg_weight', 0.0))



        lambda_match = float(self.cfg.get('lambda_match', 1.0))
        lambda_route = float(self.cfg.get('lambda_route', 0.5))
        lambda_class_sim = float(self.cfg.get('lambda_class_sim', 0.5))

        pfc_reg_loss = loss_dict['pfc_reg_loss']

        total_loss = (
            clf_loss
            + lambda_match * loss_dict['matching_loss']
            + lambda_route * loss_dict['routing_loss']
            + lambda_class_sim * loss_dict['class_sim_loss']
            + pfc_w * pfc_reg_loss
        )

        total_loss.backward()
        if self.cfg['max_norm'] != 'None':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['max_norm'])


        for param in self.model.parameters():
            if param.grad is not None and torch.all(param.grad == 0):
                param.grad = None

        self.optimizer.step()

        return bag_preds, key_indices


