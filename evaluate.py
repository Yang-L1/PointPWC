"""
Evaluation
Author: Wenxuan Wu
Date: May 2020
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict


import cmd_args
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d


def compute_inlier_ratio(flow_pred, flow_gt, inlier_thr=0.04, s2t_flow=None):
    inlier = torch.sum((flow_pred - flow_gt) ** 2, dim=2) < inlier_thr ** 2
    IR = inlier.sum().float() /( inlier.shape[0] * inlier.shape[1])
    return IR


def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx


def blend_anchor_motion (query_loc, reference_loc, reference_flow , knn=3, search_radius=0.1) :
    '''approximate flow on query points
    this function assume query points are sub- or un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    # from datasets.utils import knn_point_np
    dists, idx = knn_point_np (knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    mask = dists>search_radius
    dists[mask] = 1e+10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    mask = mask.sum(axis=1)<3

    return blended_flow, mask

def compute_nrfmr(s_pcd, flow_pred, src_pcd_raw, sflow_raw, metric_index_list, recall_thr=0.04):




    nrfmr = 0.

    for i in range ( len(s_pcd)):

        # get the metric points' transformed position
        metric_index = metric_index_list[i]
        sflow = sflow_raw[i]
        s_pcd_raw_i = src_pcd_raw[i]
        metric_pcd = s_pcd_raw_i [ metric_index ]
        metric_sflow = sflow [ metric_index ]
        metric_pcd_wrapped_gt = metric_pcd + metric_sflow
        # metric_pcd_wrapped_gt = ( torch.matmul( batched_rot[i], metric_pcd_deformed.T) + batched_trn[i] ).T


        # use the match prediction as the motion anchor
        motion_pred = flow_pred[i]
        metric_motion_pred, valid_mask = blend_anchor_motion(
            metric_pcd.cpu().numpy(), s_pcd[i].cpu().numpy(), motion_pred.cpu().numpy(), knn=3, search_radius=0.1)
        metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)

        debug = F
        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            metric_pcd_wrapped_gt = metric_pcd_wrapped_gt.cpu()
            metric_pcd_wrapped_pred = metric_pcd_wrapped_pred.cpu()
            err = metric_pcd_wrapped_pred - metric_pcd_wrapped_gt
            mlab.points3d(metric_pcd[:, 0], metric_pcd[:, 1], metric_pcd[:, 2], scale_factor=scale_factor, color=c_red)
            mlab.points3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(metric_pcd_wrapped_pred[ :, 0] , metric_pcd_wrapped_pred[ :, 1], metric_pcd_wrapped_pred[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.quiver3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], err[:, 0], err[:, 1], err[:, 2],
                          scale_factor=1, mode='2ddash', line_width=1.)
            mlab.show()

        dist = torch.sqrt( torch.sum( (metric_pcd_wrapped_pred - metric_pcd_wrapped_gt)**2, dim=1 ) )

        r = (dist < recall_thr).float().sum() / len(dist)
        nrfmr = nrfmr + r

    nrfmr = nrfmr /len(s_pcd)

    return  nrfmr




def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])


    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('config_evaluate.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvSceneFlow()

    from datasets._4DMatch import _4DMatch
    val_dataset = _4DMatch("test")

    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    metrics = defaultdict(lambda:list())

    IR=0.
    NFMR=0.

    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, src_pcd_raw, sflow_raw, metric_index = data

        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        model = model.eval()
        with torch.no_grad(): 
            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

            loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)

            full_flow = pred_flows[0].permute(0, 2, 1)
            epe3d = torch.norm(full_flow - flow, dim = 2).mean()

        i_rate = compute_inlier_ratio(full_flow, flow,inlier_thr=0.04)
        nfmr = compute_nrfmr(pos1, full_flow,  src_pcd_raw, sflow_raw, metric_index)

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        pc1_np = pos1.cpu().numpy()
        pc2_np = pos2.cpu().numpy() 
        sf_np = flow.cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)


        IR+=i_rate
        NFMR +=nfmr


    IR = IR/len(val_loader)
    NFMR = NFMR/len(val_loader)

    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers
                       ))

    print(res_str)
    print( "IR:", IR, "NFMR:", NFMR)

    logger.info(res_str)


if __name__ == '__main__':
    main()




