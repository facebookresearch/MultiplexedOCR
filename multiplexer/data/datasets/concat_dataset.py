# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)


class MixDataset(object):
    def __init__(self, datasets, ratios):
        assert len(datasets) == len(
            ratios
        ), "Number of datasets ({}) vs ratios ({}) mismatch!".format(len(datasets), len(ratios))

        self.datasets = datasets
        weighted_ratios = [ratio * len(dataset) for (ratio, dataset) in zip(ratios, datasets)]
        sum_ratios = sum(weighted_ratios)
        self.ratios = [ratio / sum_ratios for ratio in weighted_ratios]  # normalize the ratios
        self.lengths = np.array([len(dataset) for dataset in datasets])
        self.cumulative_ratios = []
        s = 0
        for i in self.ratios[:-1]:
            s += i
            self.cumulative_ratios.append(s)

    def __len__(self):
        return self.lengths.sum()

    def __getitem__(self, item):
        i = np.random.rand()
        dataset_idx = bisect.bisect_right(self.cumulative_ratios, i)
        sample_idx = np.random.randint(self.lengths[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]
