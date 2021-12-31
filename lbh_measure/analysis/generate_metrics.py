import numpy as np
import os
import argparse
import open3d as o3d
from glob import glob
from omegaconf import OmegaConf
import torch
import copy
import pandas as pd
from torchmetrics.functional import accuracy, iou, dice_score

from lbh_measure.data import BagDataset
from lbh_measure.model_inference import main
from lbh_measure.model_builder import ModelBuilder


def run_inference(annotation_dir, input_pcd_dir, model_dir):
    config = OmegaConf.load('/Users/nikhil.k/data/dev/lbh/udaan-measure/lbh/dgcnn/conf.yml')
    config.k = 9
    config.model_path = model_dir
    model = ModelBuilder.load_from_checkpoint(config=config, checkpoint_path=config.model_path)
    model.eval()

    dataset = BagDataset(input_pcd_dir, annotation_dir, downsample_factor=0.01, return_pcd=True)
    data = []
    for index, i in enumerate(glob(annotation_dir + "/*")):
        file = i.split('/')[-1].split('.')[0]
        pcd = o3d.io.read_point_cloud(
            os.path.join(input_pcd_dir, f'{file}.pcd')
        )
        pcd = pcd.voxel_down_sample(0.01)

        input_tensor, labels, _, box_pcd = dataset[index]
        vol, width, height, depth, predicted_pcd_all, box_pred_pcd, pred_tensor = main(pcd, model, 'gpu',
                                                                          vis=False, debug_output=True)

        h, w, d = box_pcd.get_max_bound() - box_pcd.get_min_bound()
        hull_gt, _ = box_pcd.compute_convex_hull()
        try:
            hull_gt.orient_triangles()
            hull_gt.compute_vertex_normals()
            hull_vol = hull_gt.get_volume()
        except RuntimeError as e:
            # if not hull_vol:
            hull_vol = w * h * d

        hull_pred, _ = box_pred_pcd.compute_convex_hull()


        lineset_gt = o3d.geometry.LineSet.create_from_triangle_mesh(hull_gt)
        lineset_pred = o3d.geometry.LineSet.create_from_triangle_mesh(hull_pred)

        pred_t_pcd = copy.deepcopy(predicted_pcd_all).translate((1.5, 0, 0))
        pred_lineset_pcd = copy.deepcopy(lineset_pred).translate((1.5, 0, 0))
        # o3d.visualization.draw_geometries([box_pcd, lineset_gt, pred_t_pcd, pred_lineset_pcd])
        dice = dice_score(pred_tensor.squeeze(0), labels)
        iou_score = iou(pred_tensor.squeeze(0), labels)
        output = {
            'gt_h': h, 'gt_w': w, 'gt_d': d, 'gt_vol': hull_vol,
            'pred_h': height, 'pred_w': width, 'pred_d': depth, 'pred_vol': vol,
            'file': file, 'dice': dice, 'iou': iou_score
        }

        data.append(output)
        print(output)
        # pred_t_lineset = copy.deepcopy(predicted_pcd).translate((1.5, 0, 0))
        # o3d.visualization.draw_geometries([box_pcd, pred_t_lineset])
    df = pd.DataFrame(data)
    print(df.to_markdown())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from rosbag to pcd")
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="Directory with multiple user annotation files")
    parser.add_argument("--input_pcd_dir", type=str, required=True, help="Directory with Point Cloud files")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory for the torch model.")
    args = parser.parse_args()
    run_inference(args.annotation_dir, args.input_pcd_dir, args.model_dir)

