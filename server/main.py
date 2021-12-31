import datetime
import json
import os

import omegaconf
import requests
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
ERROR_COUNTER = Counter("lbh_measure_cron_error_counter",
                        "Number of times the Cron has had exceptions",
                        ["exception_count"])

uri = os.getenv("CMS_DB_URI").strip('\n')
key = os.getenv("CMS_DB_KEY").strip('\n')
cosmos_client = CosmosClientBuilder(uri=uri, key=key,
                                    database_name=constants.COSMOS_DATABASE_NAME,
                                    container_name=constants.COSMOS_CONTAINER_NAME)


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


def set_status_to_success(doc, message):
    # Calling the logistics API.
    try:
        headers = {'Authorization': 'h6v5q9xqb23ppjc7bk9975hkawrxj6wa'}
        data = {
            # "awb": "${barcode}",
            "length": doc['length'] * 10,
            "width": doc['width'] * 10,
            "height": doc['height'] * 10,
            # "weight": "${wt*1000}",
            "deviceId": doc["deviceId"],
            "real_volume": doc['volume'],
            "inscan_time": doc['createdAt'],
            "img_data": ""
        }
        # response = requests.post(constants.LOGISTICS_API_ENDPOINT, files=data, header=headers)
        # # if response.status_code != 200:
        #     message = "Logistics API call failed with exception {}".format(response.json())
        #     set_status_to_failure(doc, message)
        #     return {"message": message, "status": constants.LBH_MEASURE_STATUS_FAILURE}

        return cosmos_client.upsert_item(
            doc.update({"lbh_error": message, "status": constants.LBH_MEASURE_STATUS_FAILURE})
        )
    except Exception as e:
        set_status_to_failure(doc, "Logistics API call failed with exception {}".format(e))
        return {"message": message, "status": constants.LBH_MEASURE_STATUS_FAILURE}


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
    pcd = pcd.voxel_down_sample(0.01)
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
        "length": depth,
        "breadth": width,
        "height": height,
        "bag_url": bag_url,
        "sku_id": sku_id,
        "status": constants.LBH_MEASURE_STATUS_SUCCESS,
    }
    logger.info(f'The output values of {bag_url} is {output_doc}')
    i.update(output_doc)
    cosmos_client.upsert_item(i)
    return output_doc


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


@app.on_event("startup")
@repeat_every(seconds=60 * 60, raise_exceptions=True)  # 1 hour
def poll_and_predict():
    try:
        end_ts = int(datetime.datetime.utcnow().timestamp())
        start_ts = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        start_ts = int(start_ts.utcnow().timestamp())

        # query = f"-- SELECT * FROM r WHERE r._ts between {end_ts} and {start_ts} and NOT IS_DEFINED(r.status)"
        # Query to run on all the entires
        query = f'SELECT * FROM r WHERE NOT IS_DEFINED(r.status) or r.status="{constants.LBH_MEASURE_STATUS_TODO}"'
        parameters = []
        enable_cross_partition_query = True
        doc = [i for i in cosmos_client.read_items(query, parameters, enable_cross_partition_query)]
        for i in doc:
            start_inference(i)

    except Exception as e:
        ERROR_COUNTER.labels(e.__class__).inc()
        raise e
