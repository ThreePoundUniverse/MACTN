# -*- coding: utf-8 -*-
# """
# Created on Sat Jun 18 15:37:00 2022
# @author: Dong HUANG
# Contact: huang_dong@tju.edu.cn
# """

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from time import strftime, localtime
import pandas as pd
from pathlib import Path
from easydict import EasyDict
from utils import TRAINER_REGISTOR, Accumulator, EarlyStopping


def get_trainer(trainer_name: str, *args, **kwargs):
    if trainer_name in TRAINER_REGISTOR.registered_names():
        return TRAINER_REGISTOR.get(trainer_name)(*args, **kwargs)
    else:
        raise NotImplementedError


class TorchDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        data = torch.from_numpy(data.copy()).float()
        label = torch.tensor(label).long()
        return data, label


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing. NVIDIA implements.
    Refers: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/smoothing.py
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


@TRAINER_REGISTOR.register()
class CrossVal:
    def __init__(self, train_data, train_label, val_data, val_label, model, config, writer):
        # super().__init__(train_data, train_label, val_data, val_label, model, config, writer)
        self.config = config
        self.model = model
        train_dataset = TorchDataset(train_data, train_label)
        val_dataset = TorchDataset(val_data, val_label)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.train_bs, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.val_bs, shuffle=False, num_workers=4,
                                     drop_last=False)
        self.writer = writer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2norm)
        min_lr = self.config.get('min_lr', 1e-6)
        patience = self.config.get('scheduler_patience', 8)
        factor = self.config.get('scheduler_factor', 0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
            mode='min', factor=factor, patience=patience, verbose=False, min_lr=min_lr)
        self.scaler = GradScaler()
        self.epoch_index = 0
        self.accum_iter = self.config.get('accum_iter', 2)
        self.verbose_step = self.config.get('verbose_step', 1)
        flood_b = self.config.get('flood_b', None)
        self.flood_b = flood_b if flood_b is not None else None
        earlystop_patience = self.config.get('earlystop_patience', 15)
        min_delta = self.config.get('min_delta', 0.)
        self.earlystop_callback = EarlyStopping(mode='max', patience=earlystop_patience, min_delta=min_delta, verbose=False)
        # add soft-label / label-smoothing (removed)
        self.loss_func = LabelSmoothing(smoothing=0.1)
        self.best_state = {'acc': 0.0}
        self.save_whole_model = self.config.get('is_save_model', False)
        self.input_shape = (1, train_data.shape[-2], train_data.shape[-1])

    def init_model_n_optimizer(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.model.apply(weight_init)
        self.optimizer.state = collections.defaultdict(dict)

    def check_path(self, path):
        pl_path = Path(path)
        if pl_path.suffix == '':
            if not osp.exists(path):
                os.makedirs(path)
        else:
            if not osp.exists(pl_path.parent):
                os.makedirs(pl_path.parent)
        return path

    def to_device(self, data, label):
        self.model.to(self.device)
        self.loss_func.to(self.device)
        data_ = data.to(self.device)
        label_ = label.to(self.device)
        return data_, label_

    def save_ckpt(self, state):
        self.check_path(self.config.ckpt_dir)
        filename = osp.join(self.config.ckpt_dir, self.config.ckpt_name)
        torch.save(state, filename + ".pth.tar")

    def load_ckpt(self, path=None):
        if not path:
            path_name = osp.join(self.config.ckpt_dir, self.config.ckpt_name)
            path = path_name + ".pth.tar"
        if osp.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

    def save_best_model(self):
        self.load_ckpt()
        self.check_path(self.config.model_save_dir)
        filename = osp.join(self.config.model_save_dir, self.config.ckpt_name)
        if self.save_whole_model:
            postfix = ".pkl"
            # trace_model = torch.jit.script(self.model)
            trace_model = torch.jit.trace(self.model.eval(), torch.rand(*self.input_shape).to(self.device))
            torch.jit.save(trace_model, filename + postfix)
        else:
            postfix = ".pth.tar"
            state = {"state_dict": deepcopy(self.model.state_dict())}
            torch.save(state, filename + postfix)

    def log(self, tr_acc, tr_loss, val_acc, val_loss):
        self.history['acc'].append(tr_acc)
        self.history['loss'].append(tr_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_loss'].append(val_loss)

    def package_state(self, loss=None, acc=None):
        state = {"state_dict": deepcopy(self.model.state_dict())}
        if loss and 'loss' in self.best_state.keys():
            if loss <= self.best_state['loss']:
                self.best_state['loss'] = loss
                self.save_ckpt(state)
                return
            else:
                return
        if acc and 'acc' in self.best_state.keys():
            if acc >= self.best_state['acc']:
                self.best_state['acc'] = acc
                self.save_ckpt(state)
                return
            else:
                return
    
    def train_one_epoch(self):
        self.model.train()
        metric = Accumulator(2)
        running_loss = None
        # pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'[{strftime("%Y/%m/%d-%H:%M:%S")}] Train epoch {self.epoch_index}', leave=False)
        for step, (data, label) in pbar:
            data, label = self.to_device(data, label)

            with autocast():
                out = self.model(data)
                loss_step = self.loss_func(out, label)
                if self.flood_b is not None and self.flood_b > 0:
                    loss_step = torch.abs(loss_step - self.flood_b) + self.flood_b
                self.scaler.scale(loss_step).backward()

                if running_loss is None:
                    running_loss = loss_step.item()
                else:
                    running_loss = running_loss * .99 + loss_step.item() * .01

                batch_size = data.size(0)
                prediction = torch.argmax(out, dim=-1)
                correct_ = prediction.eq(label).sum().item()
                metric.add(batch_size, correct_)

                if ((step + 1) % self.accum_iter == 0) or ((step + 1) == len(self.train_loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if ((step + 1) % self.verbose_step == 0) or ((step + 1) == len(self.train_loader)):
                    description = f'loss: {running_loss:.4f}, acc: {metric[1]/metric[0]: .3%}'
                    pbar.set_postfix_str(description)

        if self.scheduler is not None:
            self.scheduler.step()

        train_loss, train_acc = running_loss, metric[1] / metric[0]

        return train_loss, train_acc

    def fit(self):
        self.init_model_n_optimizer()  # rnn model conflict

        with torch.enable_grad():
            train_bar = tqdm(range(self.config.epochs),
                             desc=f'[{strftime("%Y/%m/%d-%H:%M:%S")}] Train',
                             ascii=True)
            for ep_index in train_bar:
                self.epoch_index = ep_index
                train_loss, train_acc = self.train_one_epoch()
                val_loss, val_acc = self.val()
                lr = self.optimizer.param_groups[0]['lr']

                self.package_state(acc=val_acc)
                self.log(train_acc, train_loss, val_acc, val_loss)

                train_bar.set_postfix_str(
                    f'LR: {lr:.6f}, '
                    f'TR loss:{train_loss:.4f}, TR acc:{train_acc:.3%}, '
                    f'VL loss: {val_loss:.4f}, VL acc: {val_acc:.3%}')

                self.writer.add_scalar('LR/lr', lr, ep_index)
                self.writer.add_scalar('Loss/training loss', train_loss, ep_index)
                self.writer.add_scalar('Acc/training acc', train_acc, ep_index)
                self.writer.add_scalar('Loss/validation loss', val_loss, ep_index)
                self.writer.add_scalar('Acc/validation acc', val_acc, ep_index)

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                self.earlystop_callback(metric=val_acc)
                if self.earlystop_callback.early_stop is True:
                    break

        # after trained, saving model
        with torch.no_grad():
            self.save_best_model()

    def val(self):
        total_num, total_loss = 0, 0.0
        total_correct = 0.0

        with torch.no_grad():
            self.model.eval()
            for b_index, (data, label) in enumerate(self.val_loader):
                data, label = self.to_device(data, label)
                out = self.model(data)

                loss_ = self.loss_func(out, label)

                total_num += data.size(0)
                total_loss += loss_.item() * data.size(0)

                prediction = torch.argmax(out, dim=-1)
                total_correct += prediction.eq(label).sum().item()

        loss = total_loss / total_num
        acc = total_correct / total_num

        return loss, acc

    def predict(self, data):
        pseudo_label = np.zeros(data.shape[0], dtype=np.int64)
        loader = DataLoader(TorchDataset(data, pseudo_label),
                            batch_size=self.config.val_bs, shuffle=False, num_workers=4)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for b_index, (data, label) in enumerate(loader):
                data, _ = self.to_device(data, label)
                out = self.model(data)
                prediction = torch.argmax(out, dim=-1)
                preds.append(prediction.cpu().numpy())
        return np.concatenate(preds)
