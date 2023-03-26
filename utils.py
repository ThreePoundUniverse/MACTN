# -*- coding: utf-8 -*-
# """
# Created on Sat Jun 18 15:37:00 2022
# @author: Dong HUANG
# Contact: huang_dong@tju.edu.cn
# """

import copy
import os
import shutil
import time
from functools import wraps
import random
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class ModelResgistry:
    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj):
        # if name in self._obj_map:
        #     raise KeyError(
        #         'An object named "{}" was already '
        #         'registered in "{}" registry'.format(name, self._name)
        #     )
        if name not in self._obj_map:
            self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:
            # Used as a decorator
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                # print(name)
                self._do_register(name, fn_or_class)
                return fn_or_class

            return wrapper

        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )
        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())


MODEL_REGISTOR = ModelResgistry('TRAINER')
MODEL_REGISTOR_MT = ModelResgistry('MULTI_TASK')
TRAINER_REGISTOR = ModelResgistry("TRAINER")


def timer_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print('start_time:', start_time) #调用foo函数之前的处理
        # print('----------调用函数前的操作-----------')
        res = func(*args, **kwargs)  # 被装饰的函数，以下指代foo，相当于调用函数foo
        # print('----------调用函数后的操作-----------')
        stop_time = time.time()  # 调用foo函数之后的处理
        # print('stop_time:',stop_time)
        print(f'Function:@{func.__name__} run time is {stop_time - start_time}')
        return res

    return wrapper


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, mode='min', patience=5, min_delta=0, verbose=False):
        """
        :param patience: how many epochs to wait before stopping when loss/acc is
               not improving
        :param min_delta: minimum difference between new loss/acc and old loss/acc for
               new loss/acc to be considered as an improvement
        """
        assert mode in {'min', 'max', None}
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'min':
            self._min_check(min_check_val=metric)
        elif self.mode == 'max':
            self._max_check(max_check_val=metric)
        else:
            raise NotImplementedError

    def _min_check(self, min_check_val):
        if self.best_metric is None:
            self.best_metric = min_check_val
        elif self.best_metric - min_check_val > self.min_delta:
            self.best_metric = min_check_val
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_metric - min_check_val < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('INFO: Early stopping')
                self.early_stop = True

    def _max_check(self, max_check_val):
        if self.best_metric is None:
            self.best_metric = max_check_val
        elif max_check_val - self.best_metric > self.min_delta:
            self.best_metric = max_check_val
            # reset counter if validation loss improves
            self.counter = 0
        elif max_check_val - self.best_metric < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('INFO: Early stopping')
                self.early_stop = True




def print_time_stamp(info: str):
    print(f'[{time.strftime("%Y/%m/%d-%H:%M:%S")}]', info)


def reset_workpath(path: str, clear: bool = False):
    if not os.path.exists(path):
        return
    if clear:
        shutil.rmtree(path)


def save_history(his_df, save_name):
    if os.path.exists(save_name):
        log_df = pd.read_csv(save_name)
        to_save_df = pd.concat([log_df, his_df], axis=1)
    else:
        to_save_df = copy.deepcopy(his_df)
    to_save_df.to_csv(save_name, index=False)


def save_log_confusion_matrix(cm: np.ndarray, save_name):
    assert cm.ndim == 2
    cm = cm[np.newaxis, :, :]
    if os.path.exists(save_name):
        log_cm = np.load(save_name)
        to_save_cm = np.concatenate([log_cm, cm], axis=0)
    else:
        to_save_cm = cm
    np.save(save_name, to_save_cm)


class LabelNames:
    def __init__(self):
        self.lookup = {
            2: ['negative', 'positive'],
            3: ['negative', 'neutral', 'positive'],
            9: ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'amusement', 'joy', 'inspiration', 'tenderness']
        }

    def get_label_names(self, num_classes: int) -> List[str]:
        return self.lookup[num_classes]

    def get_label_kind_count(self):
        return len(self.lookup)

    def get_label_kind_keys(self):
        return self.lookup.keys()


def seed_everything(seed: int = 2022):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # add 2022.08.21
    torch.backends.cudnn.benchmark = False
    print('> SEEDING DONE')


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def save_metrics(metric_df, save_name):
    if os.path.exists(save_name):
        log_df = pd.read_csv(save_name)
        to_save_df = pd.concat([log_df, metric_df], axis=0)
    else:
        to_save_df = copy.deepcopy(metric_df)
    to_save_df.to_csv(save_name, index=False)


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