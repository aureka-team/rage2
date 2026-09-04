import asyncio

import requests
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer

from rage.config import config


@cached(
    cache=Cache.REDIS,
    endpoint=config.rage_redis_host,
    port=config.rage_redis_port,
    db=config.rage_redis_db,
    serializer=PickleSerializer(),
    key="pdf:zaratustra",
)
async def get_test_pdf() -> bytes:
    response = await asyncio.to_thread(
        requests.get,
        config.test_pdf_url,
        timeout=60,
    )
    response.raise_for_status()
    return response.content
