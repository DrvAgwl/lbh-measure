import omegaconf
import torch
import uvicorn
from udaan_common.logging import logger
from udaan_common.server import create_fast_api_server
from udaan_common.resources.cosmos.cosmos_client_builder import CosmosClientBuilder

from lbh_measure.model_inference import load_model
from lbh_measure.model_inference import main, download_file
from lbh_measure.utils.convert_rosbag_to_pcd import ConvertToPCD
from lbh_measure.utils.request_types import PredictVolumeFields
import constants

app = create_fast_api_server()


# cosmos_client = CosmosClientBuilder()

def get_model(model_type):
    config = omegaconf.OmegaConf.load("./conf.yml")
    config.model_path = constants.ML_MODEL_PATH

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if model_type == 'onnx':
        import onnxruntime as ort
        model = ort.InferenceSession(config.model_path)
    else:
        model = load_model(config)
        model = model.to(device)
        model.eval()

    return model, device


model, device = get_model(model_type='onnx')
# model, device = get_model(model_type='torch')


@app.get("/healthcheck")
def ping():
    logger.info("Pong!")
    return "Pong!"


# Complex Python Fields can be directly parsed with Json Post Request
@app.post("/predict")
def predict(fields: PredictVolumeFields):
    bag_url = fields.bag_url
    sku_id = fields.sku_id

    key = "id"
    value = fields.id
    query = f"SELECT * FROM r WHERE r.{key}=@{key}"
    parameters = [{"name": f"@{key}", "value": value}]
    enable_cross_partition_query = True
    doc = {}
    # doc = [i for i in cosmos_client.read_items(query, parameters, enable_cross_partition_query)][-1]
    bag_file_path = download_file(bag_url, "/tmp/test_bag_files/")
    pcd = ConvertToPCD(topic_names=["filtered"]).get_pcd(bag_file_path)
    # vol, width, height, depth = main(pcd, model, device, False, bag_file_path, model_type='torch')
    vol, width, height, depth = main(pcd, model, device, False, bag_file_path, model_type='onnx')
    doc.update({"volume": vol,
                "width": width, "height": height, "depth": depth,
                "bag_url": bag_url, "sku_id": sku_id})
    return doc


if __name__ == "__main__":
    # Starting Service
    uvicorn.run(app, host="0.0.0.0", port=5555)
