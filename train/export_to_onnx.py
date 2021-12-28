# Databricks notebook source

# COMMAND ----------
! pip install /dbfs/mnt/central_store/libraries/lbh_measure/lbh_measure-1.0-py3-none-any.whl --no-deps --force-reinstall

# COMMAND ----------
import torch
from omegaconf import OmegaConf

from lbh_measure.model_inference import load_model


def get_model():
    config = OmegaConf.load("./conf.yml")
    config.model_path = "dbfs:/databricks/mlflow-tracking/4058348316706843/9d4620fad22146d3b4dd2743c833fe96/artifacts/epoch=36-step=2219.ckpt"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(config)
    model = model.to(device)
    return model, device


# COMMAND ----------

model, device = get_model()
input_sample = torch.zeros((1, 9, 200))
model.eval()
y = model(input_sample)

dynamic_axes = {"input_1": [0, 2], "output_1": {0: "output_1_variable_dim_0", 1: "output_1_variable_dim_1"}}
torch.onnx.export(
    model,  # model being run
    # model input (or a tuple for multiple inputs)
    input_sample,
    # where to save the model (can be a file or file-like object)
    "/dbfs/mnt/hardware/dev/v0/prod_model.onnx",
    export_params=True,
    opset_version=11,  # the ONNX version to export the model to
    do_constant_folding=True,  # # store the trained parameter weights inside the model file
    # verbose=True,
    input_names=["input_1"],
    output_names=["output_1"],
    example_outputs=y,
    dynamic_axes=dynamic_axes,
)
