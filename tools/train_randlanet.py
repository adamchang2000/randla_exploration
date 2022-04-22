# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

from distutils.command.config import config
from tracemalloc import start
import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from lib.randlanet import PoseNet
from lib.loss import Loss
from lib.utils import setup_logger
from lib.randla_utils import randla_processing
from cfg.config import YCBConfig as Config, write_config
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ExponentialLR

import faulthandler
faulthandler.enable()

from tqdm import tqdm

import open3d as o3d

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--resume_randlanet_posenet', type=str, default = '',  help='resume Randlanet PoseNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

cfg = Config()
#no refiner 
cfg.refine_start = False

def test_randla(end_points):

    img = Image.fromarray(end_points["cropped_image"].cpu().detach().numpy().squeeze())
    img.save("test.png")

    for i in range(cfg.rndla_cfg.num_layers):
        pcld = end_points["RLA_xyz_{0}".format(i)].cpu().detach().numpy().squeeze()
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pcld)
        o3d.io.write_point_cloud("test_pcld{0}.ply".format(i), pc)


def main():
    cfg.manualSeed = 2023
    random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)

    estimator = PoseNet(cfg = cfg)
    estimator.cuda()

    if opt.resume_randlanet_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(cfg.outf, opt.resume_randlanet_posenet)))

    if cfg.old_batch_mode:
        old_batch_size = cfg.batch_size
        cfg.batch_size = 1
        cfg.image_size = -1

    optimizer = optim.Adam(estimator.parameters(), lr=cfg.lr)

    if cfg.dataset == 'ycb':
        dataset = PoseDataset_ycb('test', cfg = cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
  
    cfg.sym_list = dataset.get_sym_list()
    cfg.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), -1, cfg.num_points_mesh, cfg.sym_list))

    criterion = Loss(cfg.num_points_mesh, cfg.sym_list, cfg.use_normals, cfg.use_confidence)

    if cfg.lr_scheduler == "cyclic":
        clr_div = 6
        lr_scheduler = CyclicLR(
            optimizer, base_lr=1e-5, max_lr=3e-4,
            cycle_momentum=False,
            step_size_up=cfg.nepoch * (len(dataset) / cfg.batch_size) // clr_div,
            step_size_down=cfg.nepoch * (len(dataset) / cfg.batch_size) // clr_div,
            mode='triangular'
        )
    elif cfg.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.nepoch * (len(dataset) / cfg.batch_size))
    elif cfg.lr_scheduler == "exponential":
        lr_scheduler = ExponentialLR(optimizer, 0.9)
    else:
        lr_scheduler = None

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(cfg.log_dir):
            if ".gitignore" in log:
                continue
            os.remove(os.path.join(cfg.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, cfg.nepoch):

        faulthandler.dump_traceback_later(90 * 60) #90 minutes (catch deadlock)

        write_config(cfg, os.path.join(cfg.log_dir, "config_current.yaml"))
        logger = setup_logger('epoch%d' % epoch, os.path.join(cfg.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        estimator.train()
        optimizer.zero_grad()

        for rep in range(cfg.repeat_epoch):
            trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="training")
            for batch_id, end_points in trange:

                start_time = time.time()

                end_points_cuda = {}
                for k, v in end_points.items():
                    end_points_cuda[k] = Variable(v).cuda()

                end_points = end_points_cuda

                end_points = randla_processing(end_points, cfg)

                test_randla(end_points)

                exit()

                end_points = estimator(end_points)

                loss, dis, end_points = criterion(end_points, cfg.w, False)
                loss.backward()

                dis = dis.item()
                train_dis_avg += dis
                train_count += 1
                trange.set_postfix(dis=(train_dis_avg / train_count))

                if cfg.old_batch_mode:
                    if batch_id != 0 and batch_id % old_batch_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                if batch_id != 0 and batch_id % (len(dataloader) // (40 * (old_batch_size if cfg.old_batch_mode else 1))) == 0:
                    logger.info('Epoch {} | Batch {} | dis:{}'.format(epoch, batch_id, dis))
                    torch.save(estimator.state_dict(), '{0}/randla_pose_model_current.pth'.format(cfg.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        if lr_scheduler:
            lr_scheduler.step()

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(cfg.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()