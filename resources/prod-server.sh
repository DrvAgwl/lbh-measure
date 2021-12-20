#!/bin/bash
pip install git+https://github.com/udaan-com/lbh-measure-python-service.git@6243bcdb31df21132a1051fb4d9b8939a47f313a#egg=lbh-measure
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info