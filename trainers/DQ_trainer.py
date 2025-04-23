import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker


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

        # data_loaders
        ## oulu
        self.oulu_loader = self.data_loaders['oulu']
        self.valid_data_loader = self.oulu_loader.valid_loader
        self.do_validation = self.valid_data_loader is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.oulu_loader)
        else:
            # iteration-based training
            self.oulu_loader = inf_loop(self.oulu_loader)
            self.len_epoch = self.len_epoch
        self.log_step = int(np.sqrt(self.oulu_loader.batch_size))

        # models
        self.DQNet = self.models['DQNet']
        self.DQNetclf = self.models['DQNetclf']

        # losses
        self.l1_loss = self.losses['L1']
        self.mse_loss = self.losses['MSE']
        self.ce_loss = self.losses['CE']

        # metrics
        keys_loss = ["loss"]
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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.DQNet.train()
        self.DQNetclf.train()
        self.train_metrics.reset()
        for batch_idx, (face, depth, target) in enumerate(self.oulu_loader):
            face, depth, target = face.to(self.device), depth.to(self.device), target.to(self.device).long()

            # depth
            self.optimizer1.zero_grad()
            summap, output = self.DQNet(face)
            loss1 = self.l1_loss(output, depth)# + self.mse_loss(output, depth)
            loss1.backward()
            self.optimizer1.step()
            output_depth = output.view(output.size(0), -1).mean(1)

            # 01
            if epoch > 100:
                self.optimizer2.zero_grad()
                summap, output = self.DQNet(face)
                output_01 = self.DQNetclf(summap)
                loss2 = self.ce_loss(output_01, target)
                loss2.backward()
                self.optimizer2.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss1', loss1.item())
            self.train_metrics.update('loss2', loss2.item())
            for met in self.metrics:
                if met.__name__ == 'accuracy':
                    self.train_metrics.update(met.__name__, met(output_01, target))
                elif met.__name__ == 'auc':
                    self.train_metrics.update(met.__name__, met(output_depth, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss1: {:.6f} Loss2: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss1.item(),
                    loss2.item()
                    ))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.do_lr_scheduling:
            self.lr_scheduler1.step()
            self.lr_scheduler2.step()
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
            for batch_idx, (face, depth, target) in enumerate(self.valid_data_loader):
                face, depth, target = face.to(self.device), depth.to(self.device), target.to(self.device).long()

                summap, output = self.DQNet(face)
                loss1 = self.l1_loss(output, depth)# + self.mse_loss(output, depth)
                output_depth = output.view(output.size(0), -1).mean(1)

                output_01 = self.DQNetclf(summap)
                loss2 = self.ce_loss(output_01, target)

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'valid')
                self.valid_metrics.update('loss1', loss1.item())
                self.valid_metrics.update('loss2', loss2.item())
                for met in self.metrics:
                    if met.__name__ == 'accuracy':
                        self.valid_metrics.update(met.__name__, met(output_01, target))
                    elif met.__name__ == 'auc':
                        self.valid_metrics.update(met.__name__, met(output_depth, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, param in self.DQNet.named_parameters():
            self.writer.add_histogram(name, param, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)

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
