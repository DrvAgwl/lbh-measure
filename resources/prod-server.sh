#!/bin/bash
pip install lbh_measure-1.0-py3-none-any.whl --no-deps
apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev -y
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info