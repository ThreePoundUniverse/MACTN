# -*- coding: utf-8 -*-
# """
# Created on Sat Jun 18 15:37:00 2022
# @author: Dong HUANG
# Contact: huang_dong@tju.edu.cn
# """

# %% Import
import copy
import os
import gc
from time import strftime

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
import yaml
from easydict import EasyDict
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

import preprocess
from modellibs import models
from torchutils import get_trainer
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import (print_time_stamp, reset_workpath, save_history,
                   save_log_confusion_matrix, seed_everything, save_metrics)


# %% Data First Load
def dataFirstLoad():
    channels = CFG.get('channels', 30)
    srate = CFG.get('srate', 125)
    winlen = CFG.get('windowLength', 14)
    sub_df = pd.DataFrame()
    if not os.path.exists('processData'):
        os.mkdir('processData')
    if CFG.debug:
        subNum = 1
    else:
        subNum = CFG.all_batch

    # add preprocess sequential method object
    preproc_seq = preprocess.PreProcessSequential(CFG)

    for i in range(subNum):
        if not os.path.exists(os.path.join('processData', f'sub{i + 1}')):
            os.mkdir(os.path.join('processData', f'sub{i + 1}'))
        rawdata = joblib.load(os.path.join(DATA_DIR, f'sub{i + 1}.pkl'))
        if i < CFG.first_batch:
            print(f'> PREPROCESS CHANNEL ADJUST sub{i + 1}.pkl')
            rawdata = rawdata[channel_index, :]
        else:
            print(f'> PREPROCESS sub{i + 1}.pkl')
        rawlabel = rawdata[-1, :]
        rawdata = rawdata[:-1, :]

        rawdata = preprocess.preprocess_ref(rawdata, CFG.rerefence_type)

        trial_start = np.where(rawlabel == CFG.trial_start)
        trial_end = np.where(rawlabel == CFG.trial_end)
        sample_num = 0
        for j in range(len(label_csv.label1)):
            index_tmp = label_csv.label1[j]
            index_where = np.where(rawlabel == index_tmp)
            assert len(index_where[0]) == 1
            trial_start_tmp = trial_start[0][trial_start[0] > index_where[0]]
            trial_start_tmp = min(trial_start_tmp)
            trial_end_tmp = trial_end[0][trial_end[0] > index_where[0]]
            trial_end_tmp = min(trial_end_tmp)

            data_trial = rawdata[:, trial_start_tmp:trial_end_tmp]
            data_trial = data_trial[:, :int(
                int(data_trial.shape[1] / srate / CFG.windowStep) * srate * CFG.windowStep)]
            sample = np.arange(
                winlen, data_trial.shape[1] / srate + 1e-8, CFG.windowStep)
            sample = (sample * srate).astype(np.int64)

            input_shape = CFG.get('input_shape', (channels, srate * winlen))
            data_shape = [len(sample)] + list(input_shape)
            data_tmp = np.zeros(data_shape, dtype=np.float32)
            for data_i in range(len(sample)):
                seg_data = data_trial[:,
                                      (sample[data_i] - winlen * srate):sample[data_i]]
                preproc_data = preproc_seq(seg_data)
                data_tmp[data_i] = preproc_data
            np.save(os.path.join('processData',
                    f'sub{i + 1}', f'{label_csv.label1[j]}.npy'), data_tmp)
            sample_num = sample_num + len(sample)
        sub_df = pd.concat([sub_df, pd.DataFrame(
            {'sub': [i + 1], 'num': [sample_num]})], axis=0, ignore_index=True)
        sub_df.to_csv(f'./mid_files/sub_df_{CFG.all_batch}.csv', index=False)


# %% Data Second Load
def dataSecondLoad(sub_train, dataType):
    channels = CFG.get('channels', 30)
    srate = CFG.get('srate', 125)
    winlen = CFG.get('windowLength', 14)
    input_shape = CFG.get('input_shape', (channels, srate * winlen))

    data_shape = [sum(sub_train.num)] + list(input_shape)
    data_train = np.zeros(data_shape, dtype=np.float32)
    label_train = np.zeros(sum(sub_train.num), dtype=np.float32)
    assemble_num = 0
    subs = sub_train['sub']
    load_subj_bar = tqdm(range(len(sub_train)),
                         desc=f'[{strftime("%Y/%m/%d-%H:%M:%S")}] {dataType} Loading',
                         ascii=True)
    for j in load_subj_bar:
        for k in range(len(label_csv.label1)):
            rawdata_tmp = np.load(os.path.join(
                'processData', f'sub{subs[j]}', f'{label_csv.label1[k]}.npy'))
            rawdata_tmp = rawdata_tmp
            label_tmp = label_csv.label2[k]
            label_tmp = label_tmp.repeat(rawdata_tmp.shape[0])
            data_train[assemble_num:(
                assemble_num + rawdata_tmp.shape[0])] = rawdata_tmp
            label_train[assemble_num:(
                assemble_num + rawdata_tmp.shape[0])] = label_tmp
            assemble_num = assemble_num + rawdata_tmp.shape[0]
    data_permutation = np.random.permutation(data_train.shape[0])
    data_train = data_train[data_permutation]
    label_train = label_train[data_permutation]

    num_sample_logs = [np.sum(label_train == _) for _ in range(9)]
    min_samples = np.min(num_sample_logs)
    data_train_tmp, label_train_tmp = [], []
    for label_itor in range(9):
        idxs = np.where(label_train == label_itor)[0][0:min_samples]
        data_train_tmp.append(data_train[idxs])
        label_train_tmp.append(label_train[idxs])
    data_train = np.concatenate(data_train_tmp)
    label_train = np.concatenate(label_train_tmp)

    return data_train, label_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACTN')
    parser.add_argument('--workpath', '-W', type=str)
    parser.add_argument('--reload', '-R', action='store_true')
    args = parser.parse_args()

    with open(os.path.join(args.workpath, 'config.yaml')) as f:
        CFG = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    DATA_DIR = 'your_data_dir/data'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(CFG.device) if isinstance(
        CFG.device, int) else ','.join(str(_) for _ in CFG.device)
    torch.cuda.empty_cache()

    label_csv = pd.read_csv('index_label.csv')
    channel_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                     30, 31, 32, 18, 17, 33]
    channel_index = np.array(channel_index, dtype=np.int32)
    channel_index = channel_index - 1

    seed_everything(CFG.seed)
    if args.reload:
        dataFirstLoad()

    # parameter initialize
    model_name = CFG.get('model_name', 'none')
    channels = CFG.get('channels', 30)
    srate = CFG.get('srate', 125)
    winlen = CFG.get('windowLength', 14)
    input_shape = CFG.get('input_shape', (channels, srate * winlen))
    output_shape = CFG.get('num_classes', 9)

    trainer_name = CFG.get('trainer_name', 'CrossVal')

    CFG.ckpt_dir = f'{args.workpath}/ckpt'
    CFG.model_save_dir = f'{args.workpath}/model'
    model_metrics_dir = os.path.join(args.workpath, 'ckpt')
    if not os.path.exists(model_metrics_dir):
        os.makedirs(model_metrics_dir)

    # reset work dir
    reset_workpath(model_metrics_dir, clear=True)
    reset_workpath(CFG.model_save_dir, clear=True)

    # Train & Valid
    sub_df = pd.read_csv(f'./mid_files/sub_df_{CFG.all_batch}.csv')
    sub_per = sub_df.sample(frac=1, random_state=CFG.seed, ignore_index=True)
    for i in range(CFG.n_fold):
        sub_valid = sub_per[i * int(CFG.all_batch / CFG.n_fold):(i + 1) * int(CFG.all_batch / CFG.n_fold)].reset_index(
            drop=True)
        sub_train = sub_per[~sub_per['sub'].isin(
            sub_valid['sub'])].reset_index(drop=True)
        data_train, label_train_raw = dataSecondLoad(sub_train, 'TRAIN')
        data_valid, label_valid_raw = dataSecondLoad(sub_valid, 'VALID')

        # save model name with fold index
        CFG.ckpt_name = f'ckpt_{model_name}_{i}'

        # train logger
        writer = SummaryWriter(log_dir=CFG.ckpt_dir + f'/event/fold_{i}')

        # start training
        print_time_stamp(
            f'Information --> model: {model_name}, n_fold: {i}, WP: {os.path.basename(args.workpath)}')
        model = models.get_model(model_name, input_shape, output_shape)
        trainer = get_trainer(trainer_name, data_train, label_train_raw,
                              data_valid, label_valid_raw, model, CFG, writer)
        trainer.fit()
        history = trainer.history

        gc.collect()
        trainer.load_ckpt()
        label_pred = trainer.predict(data_valid)
        del data_valid
        gc.collect()

        # draw figures
        cm_raw = confusion_matrix(
            label_valid_raw, label_pred, normalize='true')
        cm = np.around(cm_raw * 100, decimals=1)
        labels = ['anger', 'disgust', 'fear', 'sadness', 'neutral',
                  'amusement', 'joy', 'inspiration', 'tenderness']
        cmdp = ConfusionMatrixDisplay(cm, display_labels=labels)
        plt.rcParams['figure.figsize'] = [10, 10]
        cmdp.plot(cmap=plt.cm.Reds, xticks_rotation=75,
                  colorbar=True, values_format='.1f')
        plt.savefig(os.path.join(model_metrics_dir,
                    f'ConfusionMat_fold{i}.pdf'))
        plt.close()
        acc = history['acc']
        val_acc = history['val_acc']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Train_Acc')
        plt.plot(epochs, val_acc, 'b', label='Val_Acc')
        plt.title('Train and Val Acc', fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(model_metrics_dir, f'Acc_fold{i}.pdf'))
        plt.close()
        plt.plot(epochs, loss, 'bo', label='Train_Loss')
        plt.plot(epochs, val_loss, 'b', label='Val_Loss')
        plt.title('Train and Val Loss', fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(model_metrics_dir, f'Loss_fold{i}.pdf'))
        plt.close()

        writer.close()

        try:
            his_name = os.path.join(model_metrics_dir, 'history.csv')
            his_copy = copy.deepcopy(history)
            his_rename = {k+f'_{i}': v for k, v in his_copy.items()}
            his_df = pd.DataFrame(his_rename)
            save_history(his_df, his_name)
            cm_name = os.path.join(model_metrics_dir, 'confusion_matrix.npy')
            cm_copy = copy.deepcopy(cm)
            save_log_confusion_matrix(cm, cm_name)

            acc = accuracy_score(label_valid_raw, label_pred)
            precision = precision_score(
                label_valid_raw, label_pred, average='macro')
            recall = recall_score(label_valid_raw, label_pred, average='macro')
            f1 = f1_score(label_valid_raw, label_pred, average='macro')
            metrics = {'acc': [acc], 'precision': [
                precision], 'recall': [recall], 'f1': [f1]}
            metric_df = pd.DataFrame(copy.deepcopy(metrics))
            metric_name = os.path.join(model_metrics_dir, 'metrics.csv')
            save_metrics(metric_df, metric_name)
        except:
            print('save history failure')

        if CFG.debug:
            break

    try:
        cm_data_name = os.path.join(model_metrics_dir, 'confusion_matrix.npy')
        cm_data = np.load(cm_data_name)
        cm_save_name = os.path.join(
            model_metrics_dir, f'ConfusionMat_{CFG.n_fold}_folds.pdf')
        labels = ['anger', 'disgust', 'fear', 'sadness', 'neutral',
                  'amusement', 'joy', 'inspiration', 'tenderness']
        cm_data = cm_data.mean(axis=0)
        cmdp = ConfusionMatrixDisplay(cm_data, display_labels=labels)
        plt.rcParams['figure.figsize'] = [10, 10]
        cmdp.plot(cmap=plt.cm.Reds, xticks_rotation=75,
                  colorbar=True, values_format='.1f')
        plt.savefig(cm_save_name)
        plt.close()
    except:
        print('Draw confusion matrix all fold failure')
