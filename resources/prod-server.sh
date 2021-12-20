#!/bin/bash
pip install lbh_measure-1.0-py3-none-any.whl --no-deps
apt-get install ffmpeg libsm6 libxext6  -y
apt install -y libgl1-mesa-glx
apt-get install freeglut3-dev
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info