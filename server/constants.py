#  Add any constants/environment variables here

# Get the Model from mounted model share store
# ML_MODEL_PATH = "/models/hardware/v0/epoch=15-step=79.ckpt"
# ML_MODEL_PATH = "/models/hardware/v0/test.onnx"
# ML_MODEL_PATH = "/Users/nikhil.k/Downloads/epoch=19-step=99.ckpt"
# ML_MODEL_PATH = "/Users/nikhil.k/Downloads/epoch=44-step=1349.ckpt"
ML_MODEL_PATH = "/Users/nikhil.k/epoch=17-step=539.ckpt"
# ML_MODEL_PATH = "/tmp/test.onnx"

COSMOS_DATABASE_NAME = 'lbh-measure'
COSMOS_CONTAINER_NAME = 'audit'

BLOB_CONTAINER_NAME = 'tcprofilerimages'

# STATUS
LBH_MEASURE_STATUS_SUCCESS = 'success'
LBH_MEASURE_STATUS_FAILURE = 'failure'

TOPIC_NAME = ['filtered']
BAG_URL = 'https://udprodstore.blob.core.windows.net/tcprofilerimages'
