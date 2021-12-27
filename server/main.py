import datetime
import json

import omegaconf
import torch
from fastapi_utils.tasks import repeat_every
from prometheus_client import Counter
from udaan_common.logging import logger
from udaan_common.resources.cosmos.cosmos_client_builder import CosmosClientBuilder
from udaan_common.server import create_fast_api_server

import constants
from lbh_measure.model_inference import load_model
from lbh_measure.model_inference import main, download_file
from lbh_measure.utils.convert_rosbag_to_pcd import ConvertToPCD
from lbh_measure.utils.request_types import PredictVolumeFields

app = create_fast_api_server()
ERROR_COUNTER = Counter("cron_error_counter", "Number of times the Cron has had exceptions", ["exception_count"])

cosmos_client = CosmosClientBuilder(database_name="measure", container_name="audit")


def get_model(model_type):
    config = omegaconf.OmegaConf.load("./conf.yml")
    config.model_path = constants.ML_MODEL_PATH

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if model_type == "onnx":
        import onnxruntime as ort

        model = ort.InferenceSession(config.model_path)
    else:
        model = load_model(config)
        model = model.to(device)
        model.eval()

    return model, device


# model, device = get_model(model_type='onnx')
model, device = get_model(model_type="torch")


def set_status_to_failure(doc, message):
    cosmos_client.upsert_item(doc.update({"lbh_error": message, "status": constants.LBH_MEASURE_STATUS_FAILURE}))


def start_inference(i):
    logger.info(f"Found info at {i}")
    log = json.loads(i.get("log", {}))
    bag_url = f'{constants.BAG_URL}/{log["bagStorageId"]}.bag'
    sku_id = log.get("skuId")

    bag_file_path = download_file(bag_url, "/tmp/test_bag_files/", logger)
    if not bag_file_path:
        ERROR_COUNTER.labels("bag_url_not_found").inc()
        message = f"URL not found 404: {bag_url}"
        set_status_to_failure(i, message)
        return {"message": message, "status": constants.LBH_MEASURE_STATUS_FAILURE}

    pcd = ConvertToPCD(topic_names=constants.TOPIC_NAME).get_pcd(bag_file_path)
    logger.info("Converted {}".format(bag_url))

    if not pcd:
        ERROR_COUNTER.labels("topic_not_found").inc()
        message = f"Topic not found by name {constants.TOPIC_NAME}"
        set_status_to_failure(i, message)
        return {"message": message, "status": constants.LBH_MEASURE_STATUS_FAILURE}

    vol, width, height, depth = main(pcd, model, device, False, bag_file_path, model_type="torch")
    logger.info(f"{vol}, {width}")
    # vol, width, height, depth = main(pcd, model, device, False, bag_file_path, model_type='onnx')
    output_doc = {
        "volume": vol,
        "width": width,
        "height": height,
        "depth": depth,
        "bag_url": bag_url,
        "sku_id": sku_id,
        "status": constants.LBH_MEASURE_STATUS_SUCCESS,
    }
    logger.info(f'The output values of {bag_url} is {output_doc}')
    i.update(output_doc)
    cosmos_client.upsert_item(i)
    return


@app.get("/healthcheck")
def ping():
    logger.info("Pong!")
    return "Pong!"

@app.post("/predict")
def predict(fields: PredictVolumeFields):
    bag_url = fields.bag_url
    sku_id = fields.sku_id

    key = "id"
    value = fields.id
    query = f"SELECT * FROM r WHERE r.{key}=@{key}"
    parameters = [{"name": f"@{key}", "value": value}]
    enable_cross_partition_query = True
    doc = [i for i in cosmos_client.read_items(query, parameters, enable_cross_partition_query)][-1]

    response = start_inference(doc)

    return response

# Complex Python Fields can be directly parsed with Json Post Request
@app.on_event("startup")
@repeat_every(seconds=1, raise_exceptions=True)  # 1 hour
def poll_and_predict():
    try:
        # key = "id"
        # value = "AULM7GBTZMEKSCTGG2CRD6SWWX4JT"
        start_ts = datetime.datetime.now()
        end_ts = datetime.datetime.now()
        query = f"SELECT * FROM r WHERE r._ts between 1637800431 and 1637883231"
        # query = f"SELECT * FROM r"
        # parameters = [{"name": f"@{key}", "value": value}]
        parameters = []
        enable_cross_partition_query = True
        doc = [i for i in cosmos_client.read_items(query, parameters, enable_cross_partition_query)]
        for i in doc:
            start_inference(i)

    except Exception as e:
        ERROR_COUNTER.labels(e.__class__).inc()
        raise e
