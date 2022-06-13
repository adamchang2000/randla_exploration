import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import random_rotation_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import open3d as o3d
from lib.depth_utils import compute_normals, fill_missing
import cv2
from cfg.config import Config

class YCBSemanticSegDataset(data.Dataset):
    def __init__(self, mode, cfg):

        self.cfg = cfg

        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
            self.add_noise = True
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
            self.add_noise = False #only add noise to training samples

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.front_num = 2

        print(len(self.list))

    def real_gen(self):
        n = len(self.real)
        idx = np.random.randint(0, n)
        item = self.real[idx]
        return item

    def rand_range(self, lo, hi):
        return np.random.rand()*(hi-lo)+lo

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.cfg.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.cfg.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.cfg.root, real_item+'-color.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def gaussian_noise(self, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + np.random.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = np.random
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(img, rng.randint(15))
        else:
            img = self.gaussian_noise(img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def get_item(self, index, img, depth, label, meta, return_intr=False):

        cam_scale = meta['factor_depth'][0][0]

        if self.cfg.fill_depth:
            depth = fill_missing(depth, cam_scale, 1)

        if self.cfg.blur_depth:
            depth = cv2.GaussianBlur(depth,(3,3),cv2.BORDER_DEFAULT)

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_depth
        if len(mask.nonzero()[0]) <= self.minimum_num_pt:
            return {}

        choose = mask.flatten().nonzero()[0]

        if len(choose) == 0:
            return {}

        if len(choose) > self.cfg.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.cfg.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.cfg.num_points - len(choose)), 'wrap')

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3]

        if self.list[index][:8] == 'data_syn':
            img = self.rgb_add_noise(img)
            img, depth = self.add_real_back(img, label, depth, mask_depth)
            if np.random.rand() > 0.5:
                img = self.rgb_add_noise(img)

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
        label_masked = label.flatten()[choose][:, np.newaxis].astype(np.int64)
        choose = np.array([choose])

        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise and self.cfg.noise_trans > 0:
            add_t = np.random.uniform(-self.cfg.noise_trans, self.cfg.noise_trans, (self.cfg.num_points, 3))
            cloud = np.add(cloud, add_t)

        #NORMALS
        if self.cfg.use_normals:
            depth_mm = (depth * (1000 / cam_scale)).astype(np.uint16)
            normals = compute_normals(depth_mm, cam_fx, cam_fy)
            normals_masked = normals.reshape((-1, 3))[choose].astype(np.float32).squeeze(0)

        img = np.transpose(img, (2, 0, 1))
        #[0-1]
        img_normalized = img.astype(np.float32) / 255.
        img_normalized = self.norm(torch.from_numpy(img_normalized))

        if self.cfg.use_colors:
            cloud_colors = img_normalized.view((3, -1)).transpose(0, 1)[choose]

        end_points = {}

        end_points["cloud_mean"] = torch.from_numpy(np.mean(cloud.astype(np.float32), axis=0, keepdims=True))
        end_points["cloud"] = torch.from_numpy(cloud.astype(np.float32)) - end_points["cloud_mean"]

        if self.cfg.use_normals:
            end_points["normals"] = torch.from_numpy(normals_masked.astype(np.float32))

        if self.cfg.use_colors:
            end_points["cloud_colors"] = cloud_colors

        end_points["choose"] = torch.LongTensor(choose.astype(np.int32))
        end_points["img"] = img_normalized

        end_points["gt_seg"] = torch.from_numpy(label_masked)

        if return_intr:
            if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
                cam_cx = self.cam_cx_2
                cam_cy = self.cam_cy_2
                cam_fx = self.cam_fx_2
                cam_fy = self.cam_fy_2
            else:
                cam_cx = self.cam_cx_1
                cam_cy = self.cam_cy_1
                cam_fx = self.cam_fx_1
                cam_fy = self.cam_fy_1

            end_points["intr"] = (cam_fx, cam_fy, cam_cx, cam_cy)

        return end_points


    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.cfg.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.cfg.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.cfg.root, self.list[index]))
        end_points = self.get_item(index, img, depth, label, meta)

        return end_points


    def __len__(self):
        return self.length

class YCBSemanticSegDatasetWithIntr(YCBSemanticSegDataset):
    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.cfg.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.cfg.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.cfg.root, self.list[index]))

        end_points = self.get_item(index, img, depth, label, meta, return_intr=True)
        return end_points

def convert_mesh_uvs_to_colors(mesh):
    triangle_indices = np.array(mesh.triangles)
    triangle_uvs = np.array(mesh.triangle_uvs)

    vertex_colors = np.array(mesh.vertex_colors)

    indices = triangle_indices.flatten()

    vertex_uvs_as_colors = np.zeros_like(vertex_colors)
    vertex_uvs_as_colors[:,:2][indices] = triangle_uvs

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_uvs_as_colors)

    return mesh

def uniformly_sample_mesh_with_textures_as_colors(mesh, texture, number_of_points):
    pcld = mesh.sample_points_uniformly(number_of_points=number_of_points)
    
    uvs = np.array(pcld.colors)

    texture_y, texture_x, _ = texture.shape
    texture = texture.reshape((-1, 3))

    uvs_flattened = np.floor(uvs[:,0] * texture_x + np.floor(( 1. - uvs[:,1]) * texture_y) * texture_x).astype(np.int)

    colors = (texture[uvs_flattened] / 255.).astype(np.float32)
    pcld.colors = o3d.utility.Vector3dVector(colors)

    return pcld

class YCBObjectPoints(data.Dataset):
    def __init__(self, mode, cfg):
        self.cfg = cfg

        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
            self.add_noise = True
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
            self.add_noise = False #only add noise to training samples

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}

        meshes = {}
        textures = {}

        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            obj_mesh = o3d.io.read_triangle_mesh('{0}/models/{1}/textured.obj'.format(self.cfg.root, class_input[:-1]))
            obj_mesh = convert_mesh_uvs_to_colors(obj_mesh)
            textures[class_id] = np.asarray(Image.open('{0}/models/{1}/texture_map.png'.format(self.cfg.root, class_input[:-1])))

            print("loaded", class_id)

            meshes[class_id] = obj_mesh
            class_id += 1

        for k in meshes.keys():
            mesh = meshes[k]
            texture = textures[k]

            pcld = uniformly_sample_mesh_with_textures_as_colors(mesh, texture, 10000)

            points = np.array(pcld.points)
            normals = np.array(pcld.normals)
            colors = np.array(pcld.colors)

            print(points.shape, normals.shape, colors.shape)

            self.cld[k] = {}
            self.cld[k]["points"] = points
            self.cld[k]["normals"] = normals
            self.cld[k]["colors"] = colors

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        if mode == "test":
            self.list = list(self.cld.keys())
        else:
            self.list = np.array([[i for _ in range(100)] for i in list(self.cld.keys())]).flatten()

    def __getitem__(self, index):

        item = self.cld[self.list[index]]

        cloud = item["points"]
        normals = item["normals"].astype(np.float32)
        colors = torch.from_numpy(item["colors"].astype(np.float32))

        mask = np.random.choice(10000, 3000, replace=False)
        cloud = cloud[mask]
        normals = normals[mask]
        colors = colors[mask]

        cloud_mean = np.mean(cloud.astype(np.float32), axis=0, keepdims=True)
        cloud -= cloud_mean

        if self.add_noise:
            random_rot = random_rotation_matrix()[:3,:3].astype(np.float32)
            cloud = cloud @ random_rot.T
            normals = normals @ random_rot.T             

        end_points = {}
        end_points["cloud_mean"] = torch.from_numpy(cloud_mean).contiguous()
        end_points["cloud"] = torch.from_numpy(cloud).contiguous()

        if self.cfg.use_normals:
            end_points["normals"] = torch.from_numpy(normals).contiguous()

        if self.cfg.use_colors:
            end_points["cloud_colors"] = colors.contiguous()

        end_points["gt_seg"] = torch.ones(cloud.shape[0], dtype=torch.long) * self.list[index]

        return end_points

    def __len__(self):
        return len(self.list)
        