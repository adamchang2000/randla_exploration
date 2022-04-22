import argparse
import os
import random
from sympy import E
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from lib.RandLA.RandLANet import Network as RandLANet
import lib.pytorch_utils as pt_utils

class PoseNet(nn.Module):
    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        self.num_points = cfg.num_points

        self.r_out = (pt_utils.Seq(128)
                    .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                    .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                    .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                    .conv1d(cfg.num_objects*6, bn=False, activation=None)
        )

        self.t_out = (pt_utils.Seq(128)
                    .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                    .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                    .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                    .conv1d(cfg.num_objects*3, bn=False, activation=None)
        )

        if cfg.use_confidence:
            self.c_out = (pt_utils.Seq(128)
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*1, bn=False, activation=None)
            )

        self.num_obj = cfg.num_objects
        
        self.rndla = RandLANet(cfg=cfg)

        self.cfg = cfg

    def forward(self, end_points):
        features = end_points["cloud"]
        bs = features.shape[0]

        normals = end_points["normals"]
        colors = end_points["cloud_colors"]

        features = torch.cat((features, normals, colors), dim=-1)

        end_points["RLA_features"] = features.transpose(1, 2)
        end_points = self.rndla(end_points)
        feat_x = end_points["RLA_embeddings"]

        rx = self.r_out(feat_x).view(bs, self.num_obj, 6, self.num_points)
        tx = self.t_out(feat_x).view(bs, self.num_obj, 3, self.num_points)

        if self.cfg.use_confidence:
            cx = torch.sigmoid(self.c_out(feat_x)).view(bs, self.num_obj, 1, self.num_points)

        obj = end_points["obj_idx"].unsqueeze(-1).unsqueeze(-1)
        obj_rx = obj.repeat(1, 1, rx.shape[2], rx.shape[3])
        obj_tx = obj.repeat(1, 1, tx.shape[2], tx.shape[3])
        if self.cfg.use_confidence:
            obj_cx = obj.repeat(1, 1, cx.shape[2], cx.shape[3])

        out_rx = torch.gather(rx, 1, obj_rx)[:,0,:,:]
        out_tx = torch.gather(tx, 1, obj_tx)[:,0,:,:]
        if self.cfg.use_confidence:
            out_cx = torch.gather(cx, 1, obj_cx)[:,0,:,:]

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        if self.cfg.use_confidence:
            out_cx = out_cx.contiguous().transpose(2, 1).contiguous()

        end_points["pred_r"] = out_rx
        end_points["pred_t"] = out_tx
        if self.cfg.use_confidence:
            end_points["pred_c"] = out_cx

        return end_points