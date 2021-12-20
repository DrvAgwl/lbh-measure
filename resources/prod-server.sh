#!/bin/bash
pip install lbh_measure-1.0-py3-none-any.whl
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info