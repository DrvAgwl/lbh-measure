from typing import Optional

from pydantic import BaseModel


class PredictVolumeFields(BaseModel):
    bag_url: str
    sku_id: str
