import math
import random
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import os
import pandas as pd
import scanpy as sc
import torch.nn.functional as F


class Incomplete_MultiviewDataset(Dataset):
    def __init__(self, data_list, mask_matrix, labels, num_views):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, num_views, axis=1)

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32) for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False) for v in range(self.num_views)]
        return index, data, mask


def get_mask(num_views, data_size, missing_rate):
    assert num_views >= 2
    miss_sample_num = math.floor(data_size * missing_rate)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, num_views])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(num_views)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[miss_ind[j]] = rand_v
    return mask


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def load_ml_data(args):
    filePath = args.dataset_dir_base + args.dataset_name + '.mat'
    # get file name
    originData = _check_keys(sio.loadmat(filePath, struct_as_record=False, squeeze_me=True))
    data = []
    
    for ind in range(len(originData['X'])):
        curData = np.array(originData['X'][ind])
        data.append(curData)
    labels = np.array(originData['Y'])

    args.multiview_dims = [dv.shape[1] for dv in data]
    args.num_views = len(data)
    args.class_num = len(np.unique(labels))
    args.data_size = labels.shape[0]
    args.z_dim = args.class_num

    if np.max(labels) == args.class_num:
        labels = labels - 1

    print(f'Number of views:{len(data)}\nNumber of samples:{len(labels)}\nNumber of class:{len(np.unique(labels))}')
    print(args.multiview_dims)

    return data, labels


def build_dataset(args):
    data_list, labels = load_ml_data(args)
    if args.norm_type == 'min-max':
        data_list = [MinMaxScaler().fit_transform(dv) for dv in data_list]
    elif args.norm_type == 'standard':
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.norm_type == 'normal':
        data_list = [Normalizer().fit(dv).transform(dv) for dv in data_list]
    else:
        pass

    mask = get_mask(args.num_views, args.data_size, args.missing_rate)
    data_list = [data_list[v] * mask[:, v:v + 1] for v in range(args.num_views)]
    incomplete_multiview_dataset = Incomplete_MultiviewDataset(data_list, mask, labels, args.num_views)

    return incomplete_multiview_dataset

