"""
# Code adapted from:
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""



import math
import random
import copy
from collections import defaultdict
import numpy as np

import torch
from torch.distributed import get_world_size, get_rank
from torch.utils.data import Sampler

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, pad=False, consecutive_sample=False, permutation=False, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.consecutive_sample = consecutive_sample
        self.permutation = permutation
        if pad:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        else:
            self.num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.permutation:
            indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            indices = list([x for x in range(len(self.dataset))])
        
        # add extra samples to make it evenly divisible
        if self.total_size > len(indices):
            indices += indices[:(self.total_size - len(indices))]

        # subsample
        if self.consecutive_sample:
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]
        else:
            indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_num_samples(self):
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas


class DatasetBN_DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, pad=False, consecutive_sample=False, permutation=False, num_replicas=None, rank=None, batch_size = 3):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.consecutive_sample = consecutive_sample
        self.permutation = permutation
        self.batch_size = batch_size
        if pad:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        else:
            self.num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # # permutation
        # items = []
        # # print("self.dataset: {}".format(self.dataset))
        # # print("self.dataset: {}".format(self.dataset[0]))
        # print("len self.dataset: {}".format(len(self.dataset)))
        #
        # items = [item for index,item in enumerate(self.dataset) ]
        # # for index, item in enumerate(self.dataset):
        # #     print("index: {}".format(index))
        # #     items.append(item)
        # if self.permutation:
        #     random.shuffle(items)
        # print("checking init 2")


        self.index_dic = defaultdict(list)
        self.datasetId_ = []
        for index, (img, mask, img_name, scale_float, datasetId) in enumerate(self.dataset):
            self.index_dic[datasetId].append(index)
            self.datasetId_.append(datasetId)
            # if index == 20:
            #     break
        self.datasetId_ = np.array(self.datasetId_)
        self.datasetIds = list(self.index_dic.keys())


    def __iter__(self):
        # Sort batch's data: each batch include bs_trn imgs which from same dataset
        batch_idxs_dict = defaultdict(list)
        for datasetId in self.datasetIds:
            idxs = copy.deepcopy(self.index_dic[datasetId])
            if len(idxs) < self.batch_size:
                idxs = np.random.choice(idxs, size=self.batch_size, replace=True)

            if self.permutation:
                random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.batch_size:
                    batch_idxs_dict[datasetId].append(batch_idxs)
                    batch_idxs = []

        avai_datasetIds = copy.deepcopy(self.datasetIds)
        final_idxs = []
        while len(avai_datasetIds) >= 1:
            selected_pids = random.sample(avai_datasetIds, 1)
            for datasetId in selected_pids:
                batch_idxs = batch_idxs_dict[datasetId].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[datasetId]) == 0:
                    avai_datasetIds.remove(datasetId)
        self.length = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_num_samples(self):
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas