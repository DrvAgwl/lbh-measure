# Databricks notebook source
import os
from argparse import ArgumentParser

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import mlflow
from pytorch_lightning.loggers import MLFlowLogger

from lbh_measure.model_builder import ModelBuilder

# COMMAND ----------

# Enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "64")

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

def main(config):
    model = ModelBuilder(config)
    checkpoint_callback = ModelCheckpoint(monitor="val_dice", mode='max', save_last=True, verbose=True, save_top_k=3)
    logger = TensorBoardLogger(config.base_dir, name="lightning_logs")

    trainer = Trainer(
      logger=logger,
      weights_save_path=config.base_dir,
      gpus=1,
      max_epochs=config.epochs,
      log_every_n_steps=2,
      callbacks=[checkpoint_callback])
    mlflow.end_run()
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
      trainer.fit(model)
      best_model_path = checkpoint_callback.best_model_path
      mlflow.log_artifact(best_model_path)
      
      for key, val in checkpoint_callback.best_k_models.items():
        mlflow.log_artifact(key)
      
      for key, val in config.items():
        mlflow.log_param(key, val)
    
    return model, checkpoint_callback

# COMMAND ----------

c = """model: dgcnn
cuda: True
pcd_dir: '/dbfs/mnt/central_store/artefacts/lbh_measurement/'
train_data: '/dbfs/mnt/central_store/artefacts/lbh_measurement/labels/train_labels'
test_data: '/dbfs/mnt/central_store/artefacts/lbh_measurement/labels/test_labels'
embs: 1024
dropout: 0.5
emb_dims: 1024
k: 9
lr: 1e-4
scheduler: cos
epochs: 150
exp_name: test_1
batch_size: 1
test_batch_size: 1
use_sgd: False"""

# COMMAND ----------

# config = OmegaConf.load(
#     '/Users/nikhil.k/data/dev/lbh/udaan-measure/dgcnn/training/conf.yml')
config = OmegaConf.create(c)
config.k = 9


# base_dir = os.path.join('/dbfs/mnt/central_store/artefacts/lbh_measurement/outputs', config.exp_name)
# base_dir = os.path.join('./lbh_measurement/outputs', config.exp_name)
base_dir = os.path.join('/dbfs/mnt/central_store/artefacts/lbh_measurement/outputs', config.exp_name)
config.base_dir = base_dir
os.makedirs(base_dir, exist_ok=True)
# os.makedirs(os.path.join(base_dir, config.exp_name), exist_ok=True)
# os.makedirs(base_dir + '/models/', exist_ok=True)    

model, checkpoint_callback = main(config)

# COMMAND ----------

checkpoint_callback.best_k_models

# COMMAND ----------


