from pydantic import PositiveInt, StrictStr
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    emb_model: StrictStr = "text-embedding-3-large"
    emb_dimensions: PositiveInt = 1024


api_config = APIConfig()
