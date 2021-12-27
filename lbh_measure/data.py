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
    """
    This class is the common interface for all the dataprocessing.
    The static methods can be used commonly across all workflows.
    """
    def __init__(self, pcd_dir, label_directory, output_type='tensor',
                 downsample_factor=0, return_pcd=False) -> None:
        super().__init__()
        self.pcd_dir = pcd_dir
        self.label_dir = label_directory
        self.labels = glob.glob(label_directory + "/*")
        self.downsample_factor = downsample_factor
        self.output_type = output_type
        self.return_pcd = return_pcd

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
            extent[-1] = extent[-1] + extent[-1] * 0.2
        ob = o3d.geometry.OrientedBoundingBox(center, R, extent)

        point_cloud_crop = pcd.crop(ob)
        return point_cloud_crop, ob

    @staticmethod
    def get_np_points(pcd):
        """
        From the point cloud, return the points
        :param pcd:
        :return: points, colors, normals (all np.arrays)
        """
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        pcd.estimate_normals()
        pcd.normalize_normals()
        normals = np.array(pcd.normals)

        return points, colors, normals

    @staticmethod
    def parse_data(self, file_name):
        """
        From the (LabelCloud)[https://github.com/ch-sa/labelCloud] annotated data,
        this function parses the JSON files to get the PointCloud and the relevant details.
        :param file_name:
        :return:
        """
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

    @staticmethod
    def prepare_input(pcd_points: np.array,
                      pcd_colors: np.array,
                      pcd_normals: np.array,
                      output_type='tensor', add_batch=False):
        """
        This method returns either tensors/np.arrays depending on the output type.
        Tensors are required while invoking a pytorch model.
        Np.array is required for Onnx model.
        :param pcd_points:
        :param pcd_colors:
        :param pcd_normals:
        :param output_type:
        :param add_batch:
        :return:
        """
        input_shape = (pcd_points.shape[0], 9)

        if output_type == 'tensor':
            input_tensor = torch.zeros(input_shape)
            input_tensor[:, 0:3] = torch.tensor(pcd_points)
            try:
                input_tensor[:, 3:6] = torch.tensor(pcd_colors)
            except RuntimeError:
                print("Could not find colors")
                pass
            try:
                input_tensor[:, 6:9] = torch.tensor(pcd_normals)
            except RuntimeError:
                print("Could not find normals")
                pass

            if add_batch:
                return input_tensor.unsqueeze(0)
            return input_tensor

        elif output_type == 'np':
            input_array = np.zeros(input_shape)
            input_array[0, :, 0:3] = pcd_points
            try:
                input_array[0, :, 3:6] = pcd_colors
            except RuntimeError:
                print("Could not find colors")
                pass
            try:
                input_array[0, :, 6:] = pcd_normals
            except RuntimeError:
                print("Could not find colors")
                pass

            if add_batch:
                return np.expand_dims(input_array, 0)
            return input_array
        else:
            raise InputTypeNotFoundException(
                f'The give input type {output_type} is not found. Only allowed - tensor, np'
            )

    def __getitem__(self, index):  # -> tuple(torch.tensor, torch.tensor):
        file_name = self.labels[index]

        parsed_data, pcd = self.parse_data(self, file_name)
        pcd = pcd.voxel_down_sample(self.downsample_factor)

        box_pcd, ob_box = self.crop_volume(pcd, parsed_data['cart'], 'cart')

        all_points, all_colors, all_normals = self.get_np_points(pcd)
        box_points, box_colors, box_normals = self.get_np_points(box_pcd)

        input_tensor = self.prepare_input(
            pcd_points=all_points, pcd_colors=all_colors, pcd_normals=all_normals,
            output_type=self.output_type
        )

        #         threshold = 0.01
        #         roi_pcd = roi_pcd.select_by_index(np.where(all_points[:, 2]>threshold)[0])
        #         box_pcd = box_pcd.select_by_index(np.where(box_points[:, 2]>threshold)[0])

        #         all_points, roi_colors, roi_normals = self.get_np_points(roi_pcd)
        #         box_points, box_colors, box_normals = self.get_np_points(box_pcd)

        labels_one_hot = self.get_labels(box_points, all_points)
        try:
            if self.return_pcd:
                return input_tensor, labels_one_hot, input_tensor.shape[0], box_pcd
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


class InputTypeNotFoundException(Exception):
    pass
