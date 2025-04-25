import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from base import BaseTrainer
from models.metric import MetricTracker
from utils import inf_loop, consuming_time


# pretrain DQNet
class DQTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, torch_objs: dict, save_dir, resume, device, **kwargs):
        self.device = device
        super(DQTrainer, self).__init__(torch_objs, save_dir, **kwargs)

        if resume is not None:
            self._resume_checkpoint(resume)

        # datasets
        self.dataset_name = "MSU"
        # data_loaders
        self.do_validation = self.valid_data_loaders[self.dataset_name] is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loaders[self.dataset_name])
        else:
            # iteration-based training
            self.train_data_loaders[self.dataset_name] = inf_loop(self.train_data_loaders[self.dataset_name])
        self.log_step = int(np.sqrt(self.train_data_loaders[self.dataset_name].batch_size))
        self.train_step, self.valid_step = 0, 0

        # models
        self.DQNet = self.models['DQNet']
        self.DQNetclf = self.models['DQNetclf']

        # losses
        self.l1_loss = self.losses['L1']
        self.mse_loss = self.losses['MSE']
        self.ce_loss = self.losses['CE']

        # metrics
        keys_loss = ['L1_loss', 'CE_loss']
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = [m.__name__ for m in self.metrics_epoch]
        self.train_metrics = MetricTracker(
            keys_loss + keys_iter, keys_epoch, writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            keys_loss + keys_iter, keys_epoch, writer=self.writer
        )

        # optimizers
        self.optimizer1 = self.optimizers['DQNet']
        self.optimizer2 = self.optimizers['DQNetclf']

        # learning rate schedulers
        self.do_lr_scheduling = len(self.lr_schedulers) > 0
        self.lr_scheduler1 = self.lr_schedulers['DQNet']
        self.lr_scheduler2 = self.lr_schedulers['DQNetclf']

        self.clf_start_epoch = 2

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.DQNet.train()
        self.DQNetclf.train()
        self.train_metrics.reset()
        if len(self.metrics_epoch) > 0:
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)

        train_loader = self.train_data_loaders[self.dataset_name]
        start = time.time()
        for batch_idx, (face, depth, target) in enumerate(train_loader):
            face, depth, target = face.to(self.device), depth.to(self.device), target.to(self.device).long()

            # depth
            self.optimizer1.zero_grad()
            summap, output = self.DQNet(face)
            loss1 = self.l1_loss(output, depth)# + self.mse_loss(output, depth)
            loss1.backward()
            self.optimizer1.step()
            output_depth = output.view(output.size(0), -1).mean(1)
            if len(self.metrics_epoch) > 0:
                outputs = torch.cat((outputs, output_depth))
                targets = torch.cat((targets, target))

            self.train_step += 1
            self.writer.set_step(self.train_step)
            self.train_metrics.iter_update("L1_loss", loss1.item())

            # 01
            if epoch > self.clf_start_epoch:
                self.optimizer2.zero_grad()
                summap, output = self.DQNet(face)
                output_01 = self.DQNetclf(summap)
                loss2 = self.ce_loss(output_01, target)
                loss2.backward()
                self.optimizer2.step()

                self.train_metrics.iter_update("CE_loss", loss2.item())

            for met in self.metrics_iter:
                self.train_metrics.iter_update(met.__name__, met(target, output_depth))

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ", ".join(
                    f"{key}: {value:.6f}" for key, value in current_metrics.items()
                )
                self.logger.debug(epoch_debug + metrics_debug)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        end = time.time()

        for met in self.metrics_epoch:
            self.train_metrics.epoch_update(met.__name__, met(targets, outputs))

        train_log = self.train_metrics.result()

        if self.do_validation:
            valid_log = self._valid_epoch(epoch)
            valid_log.set_index("val_" + valid_log.index.astype(str), inplace=True)

        if self.do_lr_scheduling:
            self.lr_scheduler1.step()
            if epoch > self.clf_start_epoch:
                self.lr_scheduler2.step()

        log = pd.concat([train_log, valid_log])
        epoch_log = {
            "epochs": epoch,
            "iterations": self.len_epoch * epoch,
            "Runtime": consuming_time(start, end),
        }
        epoch_info = ", ".join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}"
        self.logger.info(logger_info)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.DQNet.eval()
        self.DQNetclf.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)

            valid_loader = self.valid_data_loaders[self.dataset_name]
            for batch_idx, (face, depth, target) in enumerate(valid_loader):
                face, depth, target = face.to(self.device), depth.to(self.device), target.to(self.device).long()

                summap, output = self.DQNet(face)
                loss1 = self.l1_loss(output, depth)# + self.mse_loss(output, depth)
                output_depth = output.view(output.size(0), -1).mean(1)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output_depth))
                    targets = torch.cat((targets, target))

                self.valid_step += 1
                self.writer.set_step(self.valid_step, "valid")
                self.valid_metrics.iter_update("L1_loss", loss1.item())

                if epoch > self.clf_start_epoch:
                    output_01 = self.DQNetclf(summap)
                    loss2 = self.ce_loss(output_01, target)

                    self.valid_metrics.iter_update("CE_loss", loss2.item())

                for met in self.metrics_iter:
                    self.valid_metrics.iter_update(met.__name__, met(target, output_depth))

            for met in self.metrics_epoch:
                self.valid_metrics.epoch_update(met.__name__, met(targets, outputs))

            if self.metrics_threshold is not None:
                self.threshold = self.metrics_threshold(targets, outputs)
                self.logger.info(f"Threshold: {self.threshold}")

        valid_log = self.valid_metrics.result()

        return valid_log

    def _progress(self, batch_idx):
        ratio = "[{}/{} ({:.0f}%)]"
        return ratio.format(
            batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch
        )

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        #self.start_epoch = checkpoint['epoch'] + 1
        #self.mnt_best = checkpoint['monitor_best']

        # load each model params from checkpoint.
        self.models['DQNet'].load_state_dict(checkpoint['models']['DQNet'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
