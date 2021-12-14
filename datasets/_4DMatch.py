import os, sys, glob, torch
# sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
HMN_intrin = np.array( [443, 256, 443, 250 ])
cam_intrin = np.array( [443, 256, 443, 250 ])



class _4DMatch(Dataset):

    def __init__(self, split, data_augmentation=False):
        super(_4DMatch, self).__init__()

        assert split in ['train','val','test']

        data_root = "/home/liyang/dataset/4DMatch"

        datasplit= {
            "train": "split/train",
            "val": "split/val",
            "test": "split/4DMatch"
        }


        self.entries = self.read_entries(  datasplit[split] , data_root, d_slice=None )


        self.augment_noise =  0.002

        self.num_points = 8192

        self.cache = {}
        self.cache_size = 30000


    def read_entries (self, split, data_root, d_slice=None, shuffle= False):
        entries = glob.glob(os.path.join(data_root, split, "*/*.npz"))
        if shuffle:
            random.shuffle(entries)
        if d_slice:
            return entries[:d_slice]
        return entries


    def __len__(self):
        return len(self.entries )


    def __getitem__(self, index, debug=False):


        if index in self.cache:
            entry = self.cache[index]

        else :
            entry = np.load(self.entries[index])
            if len(self.cache) < self.cache_size:
                self.cache[index] = entry


        # get transformation
        rot = entry['rot']
        trans = entry['trans']
        s2t_flow = entry['s2t_flow']
        src_pcd = entry['s_pc']
        tgt_pcd = entry['t_pc']
        correspondences = entry['correspondences'] # obtained with search radius 0.015 m
        src_pcd_deformed = src_pcd + s2t_flow
        if "metric_index" in entry:
            metric_index = entry['metric_index'].squeeze()
        else:
            metric_index = np.array([0])


        if (trans.ndim == 1):
            trans = trans[:, None]


        src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
        tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise


        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)
        sflow = (np.matmul(rot, (src_pcd + s2t_flow).T ) + trans ).T - src_pcd

        src_pcd, tgt_pcd, sflow = self.subsample(src_pcd, tgt_pcd,sflow)


        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            mlab.points3d(src_pcd[ :, 0] , src_pcd[ :, 1], src_pcd[:,  2], scale_factor=scale_factor , color=c_red)
            mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.quiver3d(src_pcd[:, 0], src_pcd[:, 1], src_pcd[:, 2], sflow[:, 0], sflow[:, 1], sflow[:, 2],
                          scale_factor=1, mode='2ddash', line_width=1.)
            mlab.show()

        return src_pcd, tgt_pcd, src_pcd.copy(), tgt_pcd.copy(), sflow, metric_index

    def subsample(self, pc1, pc2, sflow):
        indice1 = np.arange(pc1.shape[0])
        indice2 = np.arange(pc2.shape[0])
        sampled_indices1 = np.random.choice(indice1, size=self.num_points, replace=self.num_points >= pc1.shape[0], p=None)
        sampled_indices2 = np.random.choice(indice2, size=self.num_points, replace=self.num_points >= pc2.shape[0], p=None)
        pc1 = pc1[sampled_indices1]
        pc2 = pc2[sampled_indices2]
        sflow = sflow[sampled_indices1]
        return pc1, pc2, sflow

if __name__ == '__main__':

    from easydict import EasyDict as edict

    config = { "split": "train",
               "data_root": "/home/liyang/dataset/4DMatch",
               }

    config = edict(config)

    D = _4DMatch( "train")

    D.__getitem__(10, debug=True)
