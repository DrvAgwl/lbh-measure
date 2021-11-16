from typing import Optional

from fastapi import Form
from udaan_common.logging import logger
from udaan_common.server import create_fast_api_server

from utils.request_types import DemoForm

# Handles creation of API server, enables prometheus metrics at /metrics endpoint, Instrumentation & Other Monitoring
# going forward
# Always recommended to use this against to custom server
app = create_fast_api_server()


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
def form_post(field1: DemoForm):
    return {"name": field1.name, "email": field1.email}


@app.get("/demo-get/{param}")
def demo_get(param: str, q: Optional[str] = None):
    return {"param": param, "q": q}


# If your code uses async / await, use async def:
# Example is shown below
@app.get("/demo-async-get")
async def demo_async_get():
    return {"Hello": "World"}
