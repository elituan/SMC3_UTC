from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import time
import torch
import numpy as np

from apex import amp
from runx.logx import logx
from config import assert_and_infer_cfg, update_epoch, cfg
from utils.misc import AverageMeter, prep_experiment, eval_metrics
from utils.misc import ImageDumper
from utils.trnval_utils import eval_minibatch, validate_topn
from loss.utils import get_loss
from loss.optimizer import get_optimizer, restore_opt, restore_net
import torch.distributed

import datasets
import network
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetBasedTrainer():
    def __init__(self, net, optim):
        # super().__init__(net, optim)
        self.net = net
        self.optim = optim

    def _parse_data(self, inputs):
        images, gts, _img_name, scale_float, datasetId = inputs
        self.images = images.cuda()
        self.gts = gts.cuda()
        # self._img_names = _img_name.cuda()
        self.scale_floats = scale_float.cuda()
        print ("datasetId: {}".format(datasetId))
        print ("type datasetId: {}".format(type(datasetId)))
        self.datasetIds = torch.Tensor(np.array(datasetId)).cuda()

    def _ogranize_data(self):
        unique_datasetIds = torch.unique(self.datasetIds).cpu().numpy()
        reorg_imgs = []
        reorg_gts = []
        reorg_scale_floats = []
        for current_datasetid in unique_datasetIds:
            current_datasetid = (self.datasetIds == current_datasetid).nonzero().view(-1)
            if current_datasetid.size(0) > 1:
                images = torch.index_select(self.images, index=current_datasetid, dim=0)
                gts = torch.index_select(self.gts, index=current_datasetid, dim=0)
                scale_floats = torch.index_select(self.scale_floats, index=current_datasetid, dim=0)
                reorg_imgs.append(images)
                reorg_gts.append(gts)
                reorg_scale_floats.append(scale_floats)

        # Sort the list for our modified data-parallel
        # This process helps to increase efficiency when utilizing multiple GPUs
        # However, our experiments show that this process slightly decreases the final performance
        # You can enable the following process if you prefer
        # sort_index = [x.size(0) for x in reorg_pids]
        # sort_index = [i[0] for i in sorted(enumerate(sort_index), key=lambda x: x[1], reverse=True)]
        # reorg_data = [reorg_data[i] for i in sort_index]
        # reorg_pids = [reorg_pids[i] for i in sort_index]
        # ===== The end of the sort process ==== #
        self.images = reorg_imgs
        self.gts = reorg_gts
        self.scale_floats = reorg_scale_floats


    def train(self, curr_epoch, train_loader, args):
        self.net.train()

        train_main_loss = AverageMeter()
        start_time = None
        warmup_iter = 10

        for i, data in enumerate(train_loader):
            if i <= warmup_iter:
                start_time = time.time()
            # inputs = (bs,3,713,713)
            # gts    = (bs,713,713)
            self._parse_data(data)
            self._ogranize_data()

            print("img_dataset: {}".format(self.datasetIds.cpu()))
            _img_size = self.images.size.cpu()
            batch_pixel_size = _img_size(0) * _img_size(2) * _img_size(3)
            images, gts, scale_float = self.images, self.gts, self.scale_floats
            inputs = {'images': images, 'gts': gts}

            # images, gts, _img_name, scale_float, datasetId = data
            # print("img_dataset: {}".format(datasetId))
            # batch_pixel_size = images.size(0) * images.size(2) * images.size(3)
            # images, gts, scale_float = images.cuda(), gts.cuda(), scale_float.cuda()
            # inputs = {'images': images, 'gts': gts}

            self.optim.zero_grad()


            main_loss = self.net(inputs)

            if args.apex:
                log_main_loss = main_loss.clone().detach_()
                torch.distributed.all_reduce(log_main_loss,
                                             torch.distributed.ReduceOp.SUM)
                log_main_loss = log_main_loss / args.world_size
            else:
                main_loss = main_loss.mean()
                log_main_loss = main_loss.clone().detach_()

            train_main_loss.update(log_main_loss.item(), batch_pixel_size)
            if args.fp16:
                with amp.scale_loss(main_loss, self.optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                main_loss.backward()

            self.optim.step()

            if i >= warmup_iter:
                curr_time = time.time()
                batches = i - warmup_iter + 1
                batchtime = (curr_time - start_time) / batches
            else:
                batchtime = 0

            msg = ('[epoch {}], [iter {} / {}], [train main loss {:0.6f}],'
                   ' [lr {:0.6f}] [batchtime {:0.3g}]')
            msg = msg.format(
                curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
                self.optim.param_groups[-1]['lr'], batchtime)
            logx.msg(msg)

            metrics = {'loss': train_main_loss.avg,
                       'lr': self.optim.param_groups[-1]['lr']}
            curr_iter = curr_epoch * len(train_loader) + i
            logx.metric('train', metrics, curr_iter)

            if i >= 10 and args.test_mode:
                del data, inputs, gts
                return
            del data

