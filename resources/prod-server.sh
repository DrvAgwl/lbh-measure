#!/bin/bash
pip install lbh_measure-1.0-py3-none-any.whl --no-deps
apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev -y
pip install https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-0.14.1-cp37-cp37m-manylinux_2_27_x86_64.whl
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info