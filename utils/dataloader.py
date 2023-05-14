""" Pytorch Dataloader """


import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from sklearn import model_selection
from torch.utils.data import Subset
from collections import Counter

# =============================================================================
# =============================================================================
def get_weights(ds):
    """Do one pass on dataset to get weights"""
    y = np.empty(len(ds), dtype=int)
    for idx in range(len(ds)):
        y[idx] = ds[idx][1]
    weights = np.empty(len(y))
    counts = Counter(y)
    for this_y, this_count in counts.items():
        weights[y == this_y] = 1. / this_count
    return weights

# =============================================================================
# =============================================================================
def train_test_split(dataset, n_splits, test_size=0.25):
    """Split torch dataset in train and test

    Parameters
    ----------
    dataset : instance of Dataset
        The dataset to split
    test_size : float
         should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test.

    Returns
    -------
    ds_train : instance of Dataset
        The training data.
    ds_test : instance of Dataset
        The testing data.
    """
    X = range(len(dataset))
    train_idx, test_idx = \
        next(model_selection.ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0).split(X))
    return Subset(dataset, train_idx), Subset(dataset, test_idx)
# =============================================================================
# =============================================================================
class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but gives an extra
    method for querying the subjects' structure (subjects index which dataset
    each sample comes from)
    """
    def get_subids(self):
        """Return the subjects index of each sample
        Returns
        -------
        subids : array of int, shape (n_samples,)
            The subjects indices.
        """
        subids = [k * np.ones(len(d)) for k, d in enumerate(self.datasets)]
        return np.concatenate(subids)
# =============================================================================
# =============================================================================
class SegmentDataset(Dataset):
    """Class to expose an numpy array as PyTorch dataset

    Parameters
    ----------
    seg_data : 2d array, shape (n_times, n_channels)
        The segment data.
    labels : array of int, shape (n_times,)
        The labels by segment.
    sub_ids : array of int, shape (n_times,)
        The sub_ids by segment.        
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """
    def __init__(self, seg_data, labels, sub_ids, transform=None):
        assert len(seg_data) == len(labels)
        self.seg_data = seg_data
        self.labels = labels
        self.subids = sub_ids
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X, y, z = self.seg_data[idx], self.labels[idx], self.subids[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X)
        return X, y, z