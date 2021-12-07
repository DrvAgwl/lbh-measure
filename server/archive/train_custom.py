# Databricks notebook source
# Enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "64")

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

import argparse
import os
import time


import numpy as np
import open3d as o3d
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

# COMMAND ----------

# MAGIC %run ./data

# COMMAND ----------

# MAGIC %run ./model

# COMMAND ----------

# MAGIC %run ./util

# COMMAND ----------


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    IoUs = []
    for pred, seg in zip(pred_np, seg_np):
        pred = pred[0]
        seg = seg[0]
        I_all = np.zeros(2)
        U_all = np.zeros(2)
        for sem_idx in range(seg.shape[0]):
            for sem in range(2):
                I = np.sum(np.logical_and(pred[sem_idx] == sem, seg[sem_idx] == sem))
                U = np.sum(np.logical_or(pred[sem_idx] == sem, seg[sem_idx] == sem))
                I_all[sem] += I
                U_all[sem] += U
        if visual:
            for sem in range(2):
                if U_all[sem] == 0:
                    I_all[sem] = 1
                    U_all[sem] = 1
        IoUs.append(I_all / U_all )
    IoUs = np.array(IoUs)
    return IoUs.mean()

# COMMAND ----------

c = """model: dgcnn
cuda: True
pcd_dir: '/dbfs/mnt/central_store/artefacts/lbh_measurement/'
train_data: '/dbfs/mnt/central_store/artefacts/lbh_measurement/labels/train_labels'
test_data: '/dbfs/mnt/central_store/artefacts/lbh_measurement/labels/test_labels'
embs: 1024
dropout: 0.5
emb_dims: 1024
k: 9
lr: 1e-3
scheduler: cos
epochs: 100
exp_name: test_1
batch_size: 1
test_batch_size: 1
use_sgd: False"""

# COMMAND ----------

def train(args):
    train_loader = DataLoader(BagDataset(args.pcd_dir, args.train_data, partition='train', set_normals=False), 
                              num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(BagDataset(args.pcd_dir, args.test_data, partition='test', set_normals=False), 
                            num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        volumes = []
        for data in train_loader:
            input_tensor, seg = data[0].to(device), data[1].to(device)
            computed_vol = data[2][0].detach().numpy()
            gt_vol = data[3]
            name = data[4]
            # input_tensor = input_tensor.permute(0, 2, 1)
            batch_size = input_tensor.size()[0]
            opt.zero_grad()
            seg_pred = model(input_tensor)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            # loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            pred_box_coords = input_tensor[:, :, np.where(pred_np==1)[1]].permute(0, 2, 1)[0, :, :3].detach().cpu().numpy()
            vec_pred_box_coords = o3d.utility.Vector3dVector(pred_box_coords)
            pred_volume = o3d.geometry.OrientedBoundingBox().create_from_points(vec_pred_box_coords).volume()
            # print(pred_volume, gt_vol, computed_vol, name)
            # volumes.append(pred_volume, gt_vol, computed_vol, name)
        
        
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        # train_true_seg = np.concatenate(train_true_seg, axis=0)
        # train_pred_seg = np.concatenate(train_pred_seg, axis=0) 
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        print(outstr)
                                                                                                          
        # outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, 
        #                                                                         train_loss*1.0/count,
        #                                                                         train_acc,
        #                                                                         avg_per_class_acc)
        #                                                                                         #   np.mean(train_ious))
#         io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data in test_loader:
            input_tensor, seg = data[0].to(device), data[1].to(device)
            computed_vol = data[2][0].detach().numpy()
            gt_vol = data[3]
            name = data[4]
            # data = data.permute(0, 2, 1)
            # batch_size = input_tensor.size()[0]
            batch_size = args.batch_size
            seg_pred = model(input_tensor)

            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            # loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        # test_true_seg = np.concatenate(test_true_seg, axis=0)
        # test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        pred_box_coords = input_tensor[:, :, np.where(pred_np==1)[1]].permute(0, 2, 1)[0, :, :3].detach().cpu().numpy()
        vec_pred_box_coords = o3d.utility.Vector3dVector(pred_box_coords)
#         pred_volume = o3d.geometry.OrientedBoundingBox().create_from_points(vec_pred_box_coords).volume()
        # print(pred_volume, gt_vol, computed_vol, name)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        print(outstr)
        # outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
        #                                                                                       test_loss*1.0/count,
        #                                                                                       test_acc,
        #                                                                                       avg_per_class_acc)
                                                                     
#         io.cprint(outstr)
#         torch.save(model.state_dict(), args.base_dir + '/models/model.t7')
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), args.base_dir + '/models/model.t7')

# if __name__ == '__main__':
config = OmegaConf.create(c)
config.k = 9
base_dir = os.path.join('/dbfs/mnt/central_store/artefacts/lbh_measurement/outputs', config.exp_name)
config.base_dir = base_dir
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, config.exp_name), exist_ok=True)
os.makedirs(base_dir + '/models/', exist_ok=True)    
# io.cprint(str(config))
# if not args.eval:
#     train(args, io)
# else:
#     test(args, io)

train(config)

# COMMAND ----------


