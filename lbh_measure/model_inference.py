import copy
import os

import numpy as np
import open3d as o3d
import requests
import torch
from omegaconf import OmegaConf
import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm
from lbh_measure.utils.convert_rosbag_to_pcd import ConvertToPCD

from lbh_measure.data import BagDataset
from lbh_measure.model_builder import ModelBuilder

from lbh_measure.utils.util import get_colors
from lbh_measure import invoke_model


def color_pcd(pred_np, pcd):
    pred_colors = np.zeros((pred_np.shape[0], 3))
    sem_seg_colors = get_colors()
    for index in range(pred_np.shape[0]):
        color = pred_np[index] + 1
        pred_colors[index, :] = np.array(sem_seg_colors[color])

    predicted_pcd = copy.deepcopy(pcd)
    predicted_pcd.colors = o3d.utility.Vector3dVector(pred_colors / 255.0)
    return predicted_pcd

def post_process(pred, pcd):
    # Post Processing
    predicted_pcd = color_pcd(pred, pcd)
    # predicted_pcd = pcd
    cl, ind = predicted_pcd.remove_statistical_outlier(nb_neighbors=1, std_ratio=1.0)
    predicted_pcd.select_by_index(ind)

    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(2, weights="distance")
    clf.fit(np.array(predicted_pcd.points), pred)
    pred_nn = clf.predict(np.array(predicted_pcd.points))

    predicted_pcd = color_pcd(pred_nn, predicted_pcd)

    threshold = 0.01
    box = predicted_pcd.select_by_index(np.where(pred_nn == 1)[0])
    box_p = np.array(box.points)

    box_filtered = box.select_by_index(np.where(box_p[:, 2] > threshold)[0])

    pcd_tree = o3d.geometry.KDTreeFlann(box_filtered)
    selected_points = []
    for point_index in range(np.array(box_filtered.points).shape[0]):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(box_filtered.points[point_index], 0.015)
        if k > 2:
            selected_points.append(point_index)
    box_filtered = box_filtered.select_by_index(selected_points)
    return box_filtered, predicted_pcd


def main(pcd, model, device='cpu', vis=False, file_name="", model_type='torch'):
    pcd_points, pcd_colors, pcd_normals = BagDataset.get_np_points(pcd)

    if model_type == 'torch':
        input_tensor = BagDataset.prepare_input(pcd_points, pcd_colors, 'tensor', add_batch=True)
        pred = invoke_model.invoke_torch_model(model, input_tensor)
        pred = pred.permute(0, 2, 1).contiguous()
        pred = pred.argmax(dim=2).cpu().detach().numpy()[0]
        # pred = pred.cpu().detach().numpy()[0]
    elif model_type == 'onnx':
        input_array = BagDataset.prepare_input(pcd_points, pcd_colors, 'np', add_batch=True)
        pred = invoke_model.invoke_onnx_model(model, input_array)[0]
        pred = pred.transpose(0, 2, 1)
        pred = pred.argmax(axis=2)[0]
    else:
        raise

    box_filtered, predicted_pcd = post_process(pred, pcd)

    try:
        hull, _ = box_filtered.compute_convex_hull()
        hull.orient_triangles()
        hull.compute_vertex_normals()
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        if vis:
            points = np.array(predicted_pcd.points)
            threshold = 0.01
            floor_removed = predicted_pcd.select_by_index(np.where(points[:, 2] > threshold)[0])
            pred_t_lineset = copy.deepcopy(floor_removed).translate((1.5, 0, 0))
            lineset_t = copy.deepcopy(lineset).translate((1.5, 0, 0))
            o3d.visualization.draw_geometries([pcd, pred_t_lineset, lineset_t], window_name=file_name)
    except RuntimeError as e:
        # pred_t_lineset = copy.deepcopy(box).translate((1.5, 0, 0))
        # o3d.visualization.draw_geometries([pred_t_lineset], window_name='')
        print("Could not calculate volume")

    h, w, d = box_filtered.get_max_bound() - box_filtered.get_min_bound()
    try:
        hull_vol = hull.get_volume()
        print(hull_vol, w, h, d)
        return hull_vol, w, h, d
    except RuntimeError as e:
        print("Could not create a water tight mesh")
        print(e)
        hull_vol = w * h * d
        return hull_vol, w, h, d


def remove_background(pcd):
    min_bound = np.array([-0.48, -0.65, 0.02])
    max_bound = np.array([0.55, 0.65, 0.8])
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    cropped_pcd = pcd.crop(box)
    pcd_tree = o3d.geometry.KDTreeFlann(cropped_pcd)
    selected_points = []
    for point_index in range(np.array(cropped_pcd.points).shape[0]):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(cropped_pcd.points[point_index], 0.02)
        if k > 2:
            selected_points.append(point_index)
    return cropped_pcd.select_by_index(selected_points)


def download_file(url, base_path):
    local_filename = url.split("/")[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(base_path, exist_ok=True)
        path = os.path.join(base_path, local_filename)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return path


def load_model(config):
    model = ModelBuilder.load_from_checkpoint(config=config, checkpoint_path=config.model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from rosbag to pcd")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory with annotation files")
    parser.add_argument("--bag_id", type=str, required=True, help="Input directory with annotation files")
    parser.add_argument(
        "--vis", type=bool, required=True, help="Path to the training config with the hyperparams"
    )

    args = parser.parse_args()

    if args.input_dir:
        input_files = glob(f'{args.input_dir}/*.bag')
    else:
        url = '{}/{}.bag'.format(
            # 'https://ud-dev-cdn.azureedge.net/tcprofilerimages/', args.bag_id)
            # 'https://udprodstore.blob.core.windows.net/tcprofilerimages/', args.bag_id)
            "https://udprodstore.blob.core.windows.net/tcprofilerimages", args.bag_id)
        path_to_pcd = download_file(
            url, base_path='/Users/nikhil.k/data/dev/lbh/tc_data/25_11/')
        input_files = glob(f"/Users/nikhil.k/data/dev/lbh/tc_data/25_11/{args.bag_id}.bag")

    config = OmegaConf.load('/Users/nikhil.k/data/dev/lbh/udaan-measure/lbh/dgcnn/conf.yml')
    config.k = 20
    # config.model_path = "/Users/nikhil.k/data/dev/lbh/databricks_models/mlflow_best.ckpt"
    # config.model_path = "/Users/nikhil.k/data/dev/lbh/databricks_models/epoch=141-step=1561.ckpt"
    # config.model_path = "/Users/nikhil.k/Downloads/epoch=72-step=802.ckpt"
    # config.model_path = "/Users/nikhil.k/Downloads/epoch=137-step=1517.ckpt"
    # config.model_path = "/Users/nikhil.k/Downloads/epoch=19-step=99.ckpt"
    config.model_path = "/Users/nikhil.k/Downloads/epoch=44-step=1349.ckpt"
    model = ModelBuilder.load_from_checkpoint(config=config, checkpoint_path=config.model_path)
    model.eval()
    # import onnxruntime as ort
    # ML_MODEL_PATH =
    # model = ort.InferenceSession("/tmp/test.onnx")

    convert_to_pcd = ConvertToPCD(topic_names=['filtered'])

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
        # pcd = remove_background(pcd)
        # o3d.visualization.draw_geometries([pcd], window_name='')
        # o3d.visualization.draw_geometries([pcd.voxel_down_sample(0.01)], window_name='')
        pcd = pcd.voxel_down_sample(0.01)
        # continue
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # vol, width, height, depth = main(pcd, model, device, True, file_name, 'onnx')
        vol, width, height, depth = main(pcd, model, device, True, file_name)
        out['file_name'] = file_name
        out['computed_vol'] = vol
        out['width'] = width
        out['height'] = height
        out['depth'] = depth
        output.append(out)

    out_df = pd.DataFrame(output)
    print(out_df.to_markdown())
    out_df.to_csv('~/Desktop/model_output.csv')
