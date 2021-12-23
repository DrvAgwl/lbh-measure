import numpy as np
import os
import argparse
import open3d as o3d
from glob import glob
from omegaconf import OmegaConf


from lbh_measure.data import BagDataset
from lbh_measure.model_inference import main
from lbh_measure.model_builder import ModelBuilder


def run_inference(annotation_dir, input_pcd_dir):
    config = OmegaConf.load('/Users/nikhil.k/data/dev/lbh/udaan-measure/lbh/dgcnn/conf.yml')
    config.k = 20
    config.model_path = "/Users/nikhil.k/Downloads/epoch=44-step=1349.ckpt"
    model = ModelBuilder.load_from_checkpoint(config=config, checkpoint_path=config.model_path)
    model.eval()

    dataset = BagDataset(input_pcd_dir, annotation_dir, downsample_factor=0.01)
    for index, i in enumerate(glob(annotation_dir + "/*")):
        file = i.split('/')[-1].split('.')[0]
        pcd = o3d.io.read_point_cloud(
            os.path.join(input_pcd_dir, f'{file}.pcd')
        )
        pcd = pcd.voxel_down_sample(0.01)

        input_tensor, label, _ = dataset[index]
        vol, width, height, depth = main(pcd, model, 'gpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from rosbag to pcd")
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="Directory with multiple user annotation files")
    parser.add_argument("--input_pcd_dir", type=str, required=True, help="Directory with Point Cloud files")
    args = parser.parse_args()
    run_inference(args.annotation_dir, args.input_pcd_dir)

