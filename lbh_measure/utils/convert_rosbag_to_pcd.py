import argparse
import ctypes
import os
import struct
from glob import glob

import numpy as np
import open3d as o3d
import rosbag
import sensor_msgs.point_cloud2 as pc2
from tqdm import tqdm


class ConvertToPCD():
    """From ROS Bag file with the topic /merged
    This class will parse the ros bag and generate the PCD with X,Y,Z and R,G,B values.
    It will be stored as a open3d PointCloud object.
    """

    def __init__(self, topic_names: list = ['/camera_2/depth/color/points'], logger=None) -> None:
        self.topic_names = topic_names
        self.logger = logger

    def get_pcd(self, input_file):
        print("Converting file: {}".format(input_file))
        file_name = input_file.split("/")[-1].split('.')[0]

        try:
            bag = rosbag.Bag(input_file)
        except rosbag.bag.ROSBagUnindexedException as e:
            print(e)
            self.logger.error("Unindexed Bag file: {}".format(input))
            return None

        try:
            pcd = None
            for topic, msg, t in bag.read_messages(topics=self.topic_names):
                gen = pc2.read_points(msg, skip_nans=True)
                pcd = self.parse_gen(gen)
            return pcd
        except Exception as e:
            self.logger.error("Could not read the topic - {}".format(topic))
            raise e

    def parse_gen(self, gen):
        """Parse the gen file from

        Args:
            gen (): [description]

        Returns:
            Open3d.Geometry.PointCloud: PointCloud from the rosbag.
        """
        int_data = list(gen)
        try:
            # points_list = np.asarray([[x, y, z, color, _]
            #                           for (x, y, z, color, _) in int_data], dtype=np.float32)
            # Removing color details from this.
            points_list_all = np.empty((len(int_data), len(int_data[0])))
            for row_index, row in enumerate(int_data):
                points_list_all[row_index, :] = row
            points_list = points_list_all[:, :3]
        except ValueError:
            print("4 values not present in input bag file")
            points_list = np.asarray([[x, y, z]
                                      for (x, y, z) in int_data], dtype=np.float32)

        rgb_values = []
        try:
            for x in int_data:
                test = x[3]
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f', test)
                i = struct.unpack('>l', s)[0]
                # you can get back the float value by the inverse operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000) >> 16
                g = (pack & 0x0000FF00) >> 8
                b = (pack & 0x000000FF)
                rgb_values.append([r, g, b])
                # rgb_values.append([r, g, b])
        except IndexError as e:
            print('Color details not found')

        rgb_values = np.array(rgb_values, dtype=np.float32)
        # rgb_values = cv2.cvtColor(rgb_values, cv2.COLOR_BGR2RGB)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_list)
        if len(rgb_values) != 0:
            pcd.colors = o3d.utility.Vector3dVector(rgb_values / 255.0)

        return pcd


# COMMAND ----------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert from rosbag to pcd')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with all the rosbag files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory with all the rosbag files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = glob(f'{args.input_dir}/*.bag')

    convert_to_pcd = ConvertToPCD(topic_names=['filtered'])

    for read_path in tqdm(input_files):
        file_name = read_path.split("/")[-1].split('.')[0]
        pcd = convert_to_pcd.get_pcd(read_path)
        if pcd:
            pcd_file_path = os.path.join(args.output_dir, file_name + '.pcd')
            o3d.io.write_point_cloud(pcd_file_path, pcd)
            print("Saved file to {}".format(pcd_file_path))
