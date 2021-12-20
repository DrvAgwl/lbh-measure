# Databricks notebook source


# COMMAND ----------

import glob
import json
import os

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset


class BagDataset(Dataset):
    def __init__(self, pcd_dir, label_directory, partition='train', set_normals=False) -> None:
        super().__init__()
        self.pcd_dir = pcd_dir
        self.label_dir = label_directory
        self.labels = glob.glob(label_directory + "/*")
        self.set_normals = set_normals

    @staticmethod
    def crop_volume(pcd, data, type_):
        """From the input point cloud, cropping from given x,y,z, extent and rotation.
        Rotation is asumed to be in raw degrees. 

        Args:
            pcd (PointCloud): The input point cloud to crop from
            data (dict): dict having the metadata.

        Returns:
            PointCloud: Croppped Point Cloud
            OrientedBoudingBox: To plot the bbox
        """
        centroid = data["centroid"]
        dimensions = data["dimensions"]
        rotations = data["rotations"]

        center = np.array([centroid["x"], centroid["y"],
                           centroid["z"]], dtype=np.float64)
        rotations_np = np.radians(
            np.array([rotations["x"], rotations["y"], rotations["z"]], dtype=np.float64))
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle(
            rotations_np)

        extent = np.array([dimensions["length"], dimensions["width"],
                           dimensions["height"]], dtype=np.float64)
        if type_ == 'cart':
            extent = extent + extent * 0.2
        ob = o3d.geometry.OrientedBoundingBox(center, R, extent)

        point_cloud_crop = pcd.crop(ob)
        return point_cloud_crop, ob

    @staticmethod
    def get_np_points(pcd):
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        pcd.estimate_normals()
        pcd.normalize_normals()
        normals = np.array(pcd.normals)

        return points, colors, normals

    @staticmethod
    def parse_data(self, file_name):
        with open(file_name) as f:
            data = json.load(f)

        corrected_path = data["path"].split('/')
        input_file = os.path.join(self.pcd_dir, corrected_path[-1])

        pcd = o3d.io.read_point_cloud(input_file)  # Read the point cloud
        output = {}
        output["name"] = file_name.split("/")[-1]

        if len(data["objects"]) == 0:
            print("Zero objects found in {}".format(file_name))
            raise NoObjectFoundException(
                "No objects foound in {}".format(file_name))

        parsed_data = {}
        for objects in data["objects"]:
            if objects["name"] == "roi":
                parsed_data[objects["name"]] = objects
            elif objects["name"] == "cart":
                parsed_data[objects["name"]] = objects
            else:
                raise OutOfDistributionLabelException(
                    "Out of distribution label {} found in {}".format(
                        objects['name'], file_name)
                )
        return parsed_data, pcd

    @staticmethod
    def get_labels(box_points, all_points):
        filter = np.in1d(all_points, box_points)
        filter = filter.reshape(all_points.shape)

        # Setting all labels 0 - background
        labels_one_hot = np.zeros((all_points.shape[0]))
        # Points as box are set to 1.
        labels_one_hot[filter[:, 0]] = 1
        labels_one_hot[filter[:, 1]] = 1
        labels_one_hot[filter[:, 2]] = 1

        labels_one_hot = torch.Tensor(labels_one_hot).type(torch.int64)
        return labels_one_hot

    def __getitem__(self, index):  # -> tuple(torch.tensor, torch.tensor):
        file_name = self.labels[index]

        parsed_data, pcd = self.parse_data(self, file_name)

        box_pcd, ob_box = self.crop_volume(pcd, parsed_data['cart'], 'cart')

        all_points, all_colors, all_normals = self.get_np_points(pcd)
        box_points, box_colors, box_normals = self.get_np_points(box_pcd)
        
#         threshold = 0.01
#         roi_pcd = roi_pcd.select_by_index(np.where(all_points[:, 2]>threshold)[0])
#         box_pcd = box_pcd.select_by_index(np.where(box_points[:, 2]>threshold)[0])
        
#         all_points, roi_colors, roi_normals = self.get_np_points(roi_pcd)
#         box_points, box_colors, box_normals = self.get_np_points(box_pcd)
        
        labels_one_hot = self.get_labels(box_points, all_points)
        # The shape #points, 9 (3-xyz, 3-rgb, 3-normals)
        input_tensor = torch.zeros((all_points.shape[0], 9))
        input_tensor[:, 0:3] = torch.tensor(all_points)
        input_tensor[:, 3:6] = torch.tensor(all_colors)
        if self.set_normals:
            input_tensor[:, 6:9] = torch.tensor(box_normals)

        try:
            return input_tensor, labels_one_hot, input_tensor.shape[0]
        except Exception as e:
            print('FILE NAME---->', file_name)
            raise e

    def __len__(self):
        return len(self.labels)


class NoObjectFoundException(Exception):
    pass


class OutOfDistributionLabelException(Exception):
    pass

# if __name__ == '__main__':
#     train = BagDataset('/Users/nikhil.k/data/dev/lbh/converted-bag/labels/')
#     # data, label = train[-1]
#     # print(data.shape)
#     for i, j in train:
#         print(i.shape, j.shape)

# COMMAND ----------
