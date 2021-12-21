import copy
import os

import numpy as np
import open3d as o3d
import requests
import torch

from .data import BagDataset
from .model_builder import ModelBuilder
from .utils.util import get_colors
from . import invoke_model


def color_pcd(pred_np, pcd):
    pred_colors = np.zeros((pred_np.shape[0], 3))
    sem_seg_colors = get_colors()
    for index in range(pred_np.shape[0]):
        color = pred_np[index] + 1
        pred_colors[index, :] = np.array(sem_seg_colors[color])

    predicted_pcd = copy.deepcopy(pcd)
    predicted_pcd.colors = o3d.utility.Vector3dVector(pred_colors / 255.0)
    return predicted_pcd


def main(pcd, model, device, vis=False, file_name="", model_type='torch'):
    pcd_points, pcd_colors, pcd_normals = BagDataset.get_np_points(pcd)

    # The shape #points, 9 (3-xyz, 3-rgb, 3-normals)
    input_tensor = torch.zeros((1, pcd_points.shape[0], 9)).to(device)
    input_tensor[0, :, 0:3] = torch.tensor(pcd_points)
    try:
        input_tensor[0, :, 3:6] = torch.tensor(pcd_colors)
    except RuntimeError:
        print("Could not find colors")
        pass

    if model_type == 'torch':
        input_tensor = invoke_model.prepare_input(pcd_points, pcd_colors, 'tensor')
        pred = invoke_model.invoke_torch_model(model, input_tensor)
        pred = pred.permute(0, 2, 1).contiguous()
        pred = pred.argmax(dim=2).cpu().detach().numpy()[0]
        # pred = pred.cpu().detach().numpy()[0]
    elif model_type == 'onnx':
        input_array = invoke_model.prepare_input(pcd_points, pcd_colors, 'np')
        pred = invoke_model.invoke_onnx_model(model, input_array)[0]
        pred = pred.transpose(0, 2, 1)
        pred = pred.argmax(axis=2)[0]
    else:
        raise

    predicted_pcd = color_pcd(pred, pcd)
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

    try:
        hull, _ = box_filtered.compute_convex_hull()
        hull.orient_triangles()
        hull.compute_vertex_normals()
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        if vis:
            points = np.array(predicted_pcd.points)
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
