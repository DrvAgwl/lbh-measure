import omegaconf
import torch
from fastapi import Form
from udaan_common.logging import logger
from udaan_common.resources.cosmos.cosmos_client_builder import CosmosClientBuilder
from udaan_common.server import create_fast_api_server
import open3d as o3d

from model_inference import load_model
from model_inference import main, download_file
from utils.convert_rosbag_to_pcd import ConvertToPCD
from utils.request_types import PredictVolumeFields

app = create_fast_api_server()
# cosmos_client = CosmosClientBuilder()

config = omegaconf.OmegaConf.load('./conf.yml')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = load_model(config)
model = model.to(device)


@app.get("/healthcheck")
def ping():
    logger.info("Pong!")
    return "Pong!"


# Declare the body using standard Python types, thanks to Pydantic.
# Check the example below where DemoForm is custom Body Type

# Post Request with Form Type or Multipart Form Type
@app.post("/form-post")
def form_post(field1: str = Form(...)):
    return {"field_1": field1}


# Complex Python Fields can be directly parsed with Json Post Request
@app.post("/json-post")
def form_post(fields: PredictVolumeFields):

    bag_url = fields.bag_url
    sku_id = fields.sku_id

    bag_file_path = download_file(bag_url, '/tmp/test_bag_files/')
    pcd = ConvertToPCD(topic_names=["filtered"]).get_pcd(bag_file_path)
    vol, width, height, depth = main(pcd, model, device, False, bag_file_path)
    return {
        "volume": vol,
        "width": width,
        "height": height,
        "depth": depth
    }
