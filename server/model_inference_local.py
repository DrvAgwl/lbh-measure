import copy
import argparse
from glob import glob
import pandas as pd
import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf
from tqdm import tqdm 

from model_builder import ModelBuilder
from data import BagDataset
from server.utils.util import get_colors

from server.utils.convert_rosbag_to_pcd import ConvertToPCD


def color_pcd(pred, pcd):
    colors = []
    pred_np = pred#.detach().numpy()[0]
    pred_colors = np.zeros((pred_np.shape[0], 3))
    sem_seg_colors = get_colors()
    for index in range(pred_np.shape[0]):
        color = pred_np[index]+1
        pred_colors[index, :] = np.array(sem_seg_colors[color])#/255.0

    predicted_pcd = copy.deepcopy(pcd)
    predicted_pcd.colors = o3d.utility.Vector3dVector(pred_colors/255.0)
    return predicted_pcd

def generate_vis(pred, predicted_pcd, threshold=0.01):

    box = predicted_pcd.select_by_index(np.where(pred==1)[0])
    box_p = np.array(box.points)
    box_filtered = box.select_by_index(np.where(box_p[:, 2]>threshold)[0])


    # from sklearn.neighbors import LocalOutlierFactor
    # clf = LocalOutlierFactor(n_neighbors=2)
    # out_liers = clf.fit_predict(np.array(box_filtered.points))
    # box_filtered = box_filtered.select_by_index(np.where(out_liers==1)[0])


    hull, _ = box_filtered.compute_convex_hull()
    hull.orient_triangles()
    hull.compute_vertex_normals()
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(hull)

    points = np.array(predicted_pcd.points)
    x = predicted_pcd.select_by_index(np.where(points[:,2] > threshold)[0])


    # pcd_t = copy.deepcopy(predicted_pcd).translate((1.5, 0, 0))
    # pred_t = copy.deepcopy(x).translate((2.5, 0, 0))
    # box_t = copy.deepcopy(box).translate((3.5, 0, 0))
    pred_t_lineset = copy.deepcopy(x).translate((4.5, 0, 0))
    lineset_t = copy.deepcopy(lineset).translate((4.5, 0, 0))
    # o3d.visualization.draw_geometries([pcd_t, pred_t, box_t, pred_t_lineset, lineset_t, x], window_name='')
    o3d.visualization.draw_geometries([pred_t_lineset, lineset_t], window_name='')
    # o3d.visualization.draw_geometries([pred_t], window_name='')
    # return x

def main(pcd, model, vis=False, file_name=''):
    pcd_points, pcd_colors, pcd_normals = BagDataset.get_np_points(pcd)

    # The shape #points, 9 (3-xyz, 3-rgb, 3-normals)
    input_tensor = torch.zeros((1, pcd_points.shape[0], 9))
    input_tensor[0, :, 0:3] = torch.tensor(pcd_points)
    try:
        input_tensor[0, :, 3:6] = torch.tensor(pcd_colors)
    except RuntimeError:
        print("Could not find colors")
        pass
    
    input_tensor = input_tensor.permute(0, 2, 1)
    pred = model(input_tensor)
    pred_all = pred.permute(0, 2, 1).contiguous()
    pred = pred_all.max(dim=2)[1]
    
    predicted_pcd = color_pcd(pred.detach().numpy()[0], pcd)
    cl, ind = predicted_pcd.remove_statistical_outlier(nb_neighbors=1, std_ratio=1.0)
    predicted_pcd.select_by_index(ind)
    
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(2, weights='distance')
    clf.fit(np.array(predicted_pcd.points), pred[0])
    pred_nn = clf.predict(np.array(predicted_pcd.points))

    predicted_pcd = color_pcd(pred_nn, predicted_pcd)

    threshold = 0.01
    box = predicted_pcd.select_by_index(np.where(pred_nn==1)[0])
    box_p = np.array(box.points)
    
    box_filtered = box.select_by_index(np.where(box_p[:, 2]>threshold)[0])

    pcd_tree = o3d.geometry.KDTreeFlann(box_filtered)
    selected_points = []
    for point_index in range(np.array(box_filtered.points).shape[0]):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(box_filtered.points[point_index], 0.015)
        if k > 2:
            selected_points.append(point_index)
    box_filtered = box_filtered.select_by_index(selected_points)
    # o3d.visualization.draw_geometries([box_filtered], window_name='')
 
    # cl, ind = box_filtered.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)
    # box_filtered.select_by_index(ind)

    try:    
        hull, _ = box_filtered.compute_convex_hull()
        hull.orient_triangles()
        hull.compute_vertex_normals()
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        if vis:
            points = np.array(predicted_pcd.points)
            floor_removed = predicted_pcd.select_by_index(np.where(points[:,2] > threshold)[0])
            pred_t_lineset = copy.deepcopy(floor_removed).translate((1.5, 0, 0))
            lineset_t = copy.deepcopy(lineset).translate((1.5, 0, 0))
            # o3d.visualization.draw_geometries([pcd, pred_t_lineset, lineset_t], window_name=file_name)
            o3d.visualization.draw_geometries([pred_t_lineset, lineset_t], window_name=file_name)
    except RuntimeError as e:
        # points = np.array(box.points)
        # x = box.select_by_index(np.where(points[:,2] > threshold)[0])
        pred_t_lineset = copy.deepcopy(box).translate((1.5, 0, 0))
        # o3d.visualization.draw_geometries([pred_t_lineset], window_name='')
        print("Could not calculate volume")
        # raise e

    h, w, d = box_filtered.get_max_bound() - box_filtered.get_min_bound()
    try:
        hull_vol = hull.get_volume()
        return hull_vol, w, h, d
    except RuntimeError as e:
        print("Could not create a water tight mesh")
        print(e)
        return None, w, h, d

def remove_background(pcd):
    min_bound = np.array([-0.48, -0.65, 0.02])
    max_boud = np.array([0.55, 0.65, 0.8])
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound = min_bound, max_bound=max_boud)    
    cropped_pcd = pcd.crop(box)
    pcd_tree = o3d.geometry.KDTreeFlann(cropped_pcd)
    selected_points = []
    for point_index in range(np.array(cropped_pcd.points).shape[0]):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(cropped_pcd.points[point_index], 0.02)
        if k > 2:
            selected_points.append(point_index)
    return cropped_pcd.select_by_index(selected_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from rosbag to pcd")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with annotation files")
    parser.add_argument(
        "--vis", type=bool, required=True, help="Path to the training config with the hyperparams"
    )

    args = parser.parse_args()
    input_files = glob(f"{args.input_dir}/*.bag")

    config = OmegaConf.load('/Users/nikhil.k/data/dev/lbh/udaan-measure/lbh/dgcnn/conf.yml')
    config.k = 9
    # config.model_path = "/Users/nikhil.k/data/dev/lbh/databricks_models/mlflow_best.ckpt"
    # config.model_path = "/Users/nikhil.k/data/dev/lbh/databricks_models/epoch=141-step=1561.ckpt"
    # config.model_path = "/Users/nikhil.k/Downloads/epoch=72-step=802.ckpt"
    config.model_path = "/Users/nikhil.k/Downloads/epoch=137-step=1517.ckpt"
    model = ModelBuilder.load_from_checkpoint(config=config, checkpoint_path=config.model_path)


    convert_to_pcd = ConvertToPCD(topic_names=['/merged'])
    
    output = []
    for i in tqdm(input_files):
        out = {}
        print("Converting file: {}".format(i))
        file_name = i.split("/")[-1].split('.')[0]
        # if not file_name == 'SGE00J3JXMN1GR001S':
            # continue
        pcd = convert_to_pcd.get_pcd(i)
        if pcd is None:
            print("Skipping {} Bag Unindexed bag file".format(file_name))
            continue
        pcd = remove_background(pcd)
        vol, width, height, depth = main(pcd, model, args.vis, file_name)
        out['file_name'] = file_name
        out['computed_vol'] = vol
        out['width'] = width
        out['height'] = height
        out['depth'] = depth
        output.append(out)
    
    out_df = pd.DataFrame(output)
    print(out_df.to_markdown())
    out_df.to_csv('~/Desktop/model_output.csv')
        