import asyncio

from tqdm import tqdm
from joblib import hash
from more_itertools import flatten
from abc import ABC, abstractmethod
from pydantic import BaseModel, StrictStr, StrictBool

from common.cache import RedisCache


class Document(BaseModel):
    text: StrictStr
    metadata: dict = {}
    is_table: StrictBool = False


class TextLoader(ABC):
    def __init__(
        self,
        cache: RedisCache | None = None,
        max_concurrency: int = 10,
    ):
        self.cache = cache
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _get_cache_key(self, source_path: str) -> str:
        with open(source_path, "rb") as f:
            return hash(f.read())

    @abstractmethod
    async def _get_documents(self, source_path: str) -> list[Document]:
        pass

    async def _load(
        self,
        source_path,
        pb: tqdm | None = None,
    ) -> list[Document]:
        async with self.semaphore:
            documents = await asyncio.to_thread(
                self._get_documents,
                source_path=source_path,
            )

            if pb is not None:
                pb.update(1)

            return documents

    async def load(
        self,
        source_path: str,
    ) -> list[Document]:
        cache_key = self._get_cache_key(source_path=source_path)
        if self.cache is not None:
            cached_output = self.cache.load(cache_key=cache_key)
            if cached_output is not None:
                return cached_output

        documents = await self._load(source_path=source_path)
        if self.cache:
            self.cache.save(
                cache_key=cache_key,
                obj=documents,
            )

        return documents

    async def batch_load(self, source_paths: list[str]) -> list[Document]:
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self.load(source_path=source_path))
                for source_path in source_paths
            ]

        return list(flatten((t.result() for t in tasks)))
