import argparse
import json
from operator import index
import os
import copy
from glob import glob


import numpy as np
import pandas as pd
import open3d as o3d
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from model_builder import ModelBuilder
from data import BagDataset


# def load_model(config):
#     model = DGCNN_semseg(args=config)
#     model = torch.nn.DataParallel(model)
#     model.load_state_dict(torch.load(config.model_path, map_location=torch.device("cpu")))
#     return model

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

def validate_model(test_file, config, model, measured_df, output_dir):
    metrics_dict = {}
    parsed_data, overall_pcd = BagDataset.parse_data(test_file)

    roi_pcd, ob_roi = BagDataset.crop_volume(overall_pcd, parsed_data['roi'])
    box_pcd, ob_box = BagDataset.crop_volume(overall_pcd, parsed_data['box'])
    
    roi_points, roi_colors, roi_normals = BagDataset.get_np_points(roi_pcd)
    box_points, box_colors, box_normals = BagDataset.get_np_points(box_pcd)
    
    labels_one_hot = BagDataset.get_labels(box_points, roi_points).unsqueeze(0)
    # The shape #points, 9 (3-xyz, 3-rgb, 3-normals)
    input_tensor = torch.zeros((1, 9, roi_points.shape[0]))
    input_tensor[0, 0:3, :] = torch.tensor(roi_points.transpose())
    input_tensor[0, 3:6, :] = torch.tensor(roi_colors.transpose())
    if config.get("set_normals"):
        input_tensor[0, 6:9, :] = torch.tensor(roi_normals.transpose())
    
    pred = model(input_tensor)
    pred_all = pred.permute(0, 2, 1).contiguous()
    pred = pred_all.max(dim=2)[1]
    
    metrics_dict['pred'] = pred_all.squeeze(0).detach().cpu().numpy()
    metrics_dict['gt'] = labels_one_hot.squeeze(0)
    
    pred_np = pred.detach().cpu().numpy()    
    seg_np = labels_one_hot.detach().numpy()

  
    avg_per_class_acc = metrics.balanced_accuracy_score(pred_np.reshape(-1), seg_np.reshape(-1))
    metrics_dict['avg_per_class'] = avg_per_class_acc
    test_ious = calculate_sem_IoU([pred_np], [seg_np])
    metrics_dict['iou'] = test_ious
    pred_box_coords = input_tensor[:, :, np.where(pred_np==1)[1]].permute(0, 2, 1)[0, :, :3].detach().numpy()
    box = roi_pcd.select_by_index(np.where(pred==1)[1])
    box_p = np.array(box.points)
    box = box.select_by_index(np.where(box_p[:, 2]>0)[0])
    
    hull, _ = box.compute_convex_hull()
    hull.orient_triangles()
    
    try:
        pred_volume = hull.get_volume()
        metrics_dict['pred_vol'] = pred_volume
    except:
        return
    # o3d.visualization.draw_geometries([box, ob_box], window_name='')
    # ob_box = box.get_oriented_bounding_box()
    # vec_pred_box_coords = o3d.utility.Vector3dVector(pred_box_coords)
    # pred_volume = o3d.geometry.OrientedBoundingBox().create_from_points(vec_pred_box_coords).volume()
    index_number = test_file.split('/')[-1].split("_")[0]
    print("index_number----> ",index_number)
    metrics_dict['index'] = index_number
    measured_volume = measured_df[measured_df['index']==index_number]['Volume_measured'].values[0]
    metrics_dict['gt_vol'] = measured_volume
    metrics_dict['pred_vol_box'] = box_pcd.get_oriented_bounding_box().volume()
    out_string = (
        f'average per class accuracy: {avg_per_class_acc} '
        f'test_iou: {test_ious} '
        f'pred vol: {pred_volume} '
        f'measured vol: {measured_volume} '
        f'box_pcd_vol: {box_pcd.get_oriented_bounding_box().volume()} ',
        f'computed gt vol from ob_box: {ob_box.volume()}'
    )
    print(out_string)
    # print(avg_per_class_acc, test_ious, pred_volume, box_pcd.get_oriented_bounding_box().volume(), ob_box.volume())
    # outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
    #                                                                                     avg_per_class_acc,
    #                                                                                     np.mean(test_ious))
    return metrics_dict

def calculate_test_metrics(metrics_df):
    pred, seg = metrics_df['pred'].values, metrics_df['gt'].values
    pred_tensor = torch.tensor([])
    seg_tensor = torch.tensor([])
    for i, j in zip(pred, seg):
        i = torch.tensor(i).softmax(1)
        pred_tensor = torch.cat((pred_tensor, i))
        seg_tensor = torch.cat((seg_tensor, j))
    # pred_np = pred.astype(np.float64)
    # seg_np = seg.astype(np.float64)
    # pred = torch.tensor(pred_np)
    # seg = torch.tensor(seg_np)
    # average_precision = metrics.average_precision_score(pred, seg)
    
    print(pred_tensor.shape, seg_tensor.shape)
    from torchmetrics import PrecisionRecallCurve
    pr_curve = PrecisionRecallCurve(num_classes=2)
    precision, recall, thresholds = pr_curve(pred_tensor, seg_tensor)
    print(precision[1].shape, recall[1].shape)
    # print(precision, recall, thresholds)
    # disp = metrics.PrecisionRecallDisplay(precision=precision[1].detach().numpy(), recall=recall[1].detach().numpy())
    # plt.plot()
    
    plt.figure()
    plt.step(recall[1], precision[1], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from rosbag to pcd")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with annotation files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save file")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the training config with the hyperparams"
    )

    measured_df = pd.read_csv('~/Downloads/LBH Measurement - Sheet2.csv')
    args = parser.parse_args()
    input_files = glob(f"{args.input_dir}/*.json")

    config = OmegaConf.load(args.config_path)
    config.k = 9
    config.model_path = "/Users/nikhil.k/data/dev/lbh/databricks_models/no_floor.ckpt"
    model = model = ModelBuilder.load_from_checkpoint(config=config, checkpoint_path=config.model_path)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_out = []
    for i in tqdm(input_files):
        print("Validating file: {}".format(i))
        metrics_out.append(validate_model(i, config, model, measured_df, args.output_dir))
    metrics_df = pd.DataFrame(metrics_out)
    metrics_df.to_csv('./metrics.csv')
    calculate_test_metrics(metrics_df)    
