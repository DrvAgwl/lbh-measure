import os
from tqdm import tqdm
import argparse
import json
import numpy as np
import open3d as o3d
from glob import glob


class AnnotationValidation:
    """
    Annotation Directory Structure
    Base
     |--Annotator 1
     |   |-- pcd1.json
     |   |-- pcd2.json
     |--Annotator 2
     |   |-- pcd1.json
     |   |-- pcd1.jso
    """

    def __init__(self, annotation_base_dir, input_pcd_dir):
        self.annotation_base_dir = annotation_base_dir
        self.input_pcd_dir = input_pcd_dir

    def parse_annotation(self, data: dict, label='cart'):
        """
        The annotation form LabelCloud3D.
        We are finding the label as "cart"
        :param data: (dict) This is the JSON file object from the tool.
        :param label: The label to fetch annotations for. Default - cart.
        :return: centroid, rotations, dimensions
        """
        centroid = {}
        rotations = {}
        dimensions = {}
        for objects in data["objects"]:
            if objects["name"] == label:
                object_roi = data["objects"][0]
                centroid = object_roi["centroid"]
                dimensions = object_roi["dimensions"]
                rotations = object_roi["rotations"]
        return centroid, rotations, dimensions

    def create_bounding_box(self, centroid: dict, rotations: dict, dimensions: dict):
        """
        Method to create the OrientedBounding box for open3d visualiation
        :param centroid:
        :param rotations:
        :param dimensions:
        :return:
        """
        center = np.array([centroid["x"], centroid["y"], centroid["z"]], dtype=np.float64)
        rotations_np = np.radians(np.array([rotations["x"], rotations["y"], rotations["z"]], dtype=np.float64))
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle(rotations_np)

        extent = np.array([dimensions["length"], dimensions["width"], dimensions["height"]], dtype=np.float64)

        ob = o3d.geometry.OrientedBoundingBox(center, R, extent)
        return ob

    def validate_annotation(self):
        file_names = [i.split('/')[-1].split('.')[0] for i in glob(glob(f'{self.annotation_base_dir}/*')[0]+'/*.json')]
        for file in file_names:
            pcd = o3d.io.read_point_cloud(
                os.path.join(self.input_pcd_dir, f'{file}.pcd')
            )
            vis_boxes = [pcd]
            for i in glob(f'{self.annotation_base_dir}/*'):
                with open(os.path.join(i, f'{file}.json'), 'r') as f:
                    data = json.load(f)
                centroid, rotations, dimensions = self.parse_annotation(data)
                box = self.create_bounding_box(centroid, rotations, dimensions)
                vis_boxes.append(box)
            o3d.visualization.draw_geometries(vis_boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from rosbag to pcd")
    """
    Annotation Directory Structure
    Base
     |--Annotator 1
     |   |-- pcd1.json
     |   |-- pcd2.json
     |--Annotator 2
     |   |-- pcd1.json
     |   |-- pcd1.json
    """
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="Directory with multiple user annotation files")
    parser.add_argument("--input_pcd_dir", type=str, required=True, help="Directory with Point Cloud files")
    args = parser.parse_args()
    validate_annos = AnnotationValidation(args.annotation_dir, args.input_pcd_dir)
    validate_annos.validate_annotation()
