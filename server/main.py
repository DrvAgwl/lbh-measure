import omegaconf
import uvicorn
import torch
from udaan_common.logging import logger
from udaan_common.resources.cosmos.cosmos_client_builder import CosmosClientBuilder
from udaan_common.server import create_fast_api_server

from lbh_measure.model_inference import load_model
from lbh_measure.model_inference import main, download_file
from lbh_measure.utils.convert_rosbag_to_pcd import ConvertToPCD
from lbh_measure.utils.request_types import PredictVolumeFields

app = create_fast_api_server()
# cosmos_client = CosmosClientBuilder()

config = omegaconf.OmegaConf.load('../lbh_measure/conf.yml')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = load_model(config)
model = model.to(device)


@app.get("/healthcheck")
def ping():
    logger.info("Pong!")
    return "Pong!"


# Complex Python Fields can be directly parsed with Json Post Request
@app.post("/predict")
def predict(fields: PredictVolumeFields):
    bag_url = fields.bag_url
    sku_id = fields.sku_id
    bag_file_path = download_file(bag_url, '/tmp/test_bag_files/')
    pcd = ConvertToPCD(topic_names=["filtered"]).get_pcd(bag_file_path)
    vol, width, height, depth = main(pcd, model, device, False, bag_file_path)
    return {
        "volume": vol,
        "width": width,
        "height": height,
        "depth": depth,
        "bag_url": bag_url,
        "sku_id": sku_id
    }

if __name__ == "__main__":
    # Starting Service
    uvicorn.run(app, host='0.0.0.0', port=5555)