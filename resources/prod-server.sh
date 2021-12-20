#!/bin/bash
pip install lbh_measure-1.0-py3-none-any.whl --no-deps
pip install -U open3d
apt-get install libgl1
apt install -y libgl1-mesa-glx
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info