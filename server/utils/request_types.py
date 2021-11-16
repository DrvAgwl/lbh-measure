from typing import Optional

from pydantic import BaseModel


class DemoForm(BaseModel):
    name: str
    email: str
    team: Optional[str] = None
