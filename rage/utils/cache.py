from functools import lru_cache
from common.cache import RedisCache


@lru_cache(maxsize=1)
def get_redis_cache() -> RedisCache:
    return RedisCache()
