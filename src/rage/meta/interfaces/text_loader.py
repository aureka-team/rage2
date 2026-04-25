import xxhash
import joblib
import asyncio

from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod

from tqdm import tqdm  # type: ignore
from more_itertools import flatten

from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer

from pydantic import BaseModel, StrictStr, Field

from rage.config import config


def get_cache_key(
    func: Any,
    _self: Any,
    *args: Any,
    **kwargs: Any,
) -> str:
    cache_key = joblib.hash(
        (
            func.__module__,
            func.__qualname__,
            args,
            kwargs,
        )
    )

    assert cache_key is not None
    return cache_key


class Document(BaseModel):
    text: StrictStr = Field(min_length=1)
    metadata: dict = {}


class TextLoader(ABC):
    def __init__(
        self,
        max_concurrency: int = 10,
    ):

        self.semaphore = asyncio.Semaphore(max_concurrency)

    @abstractmethod
    async def get_documents(
        self,
        source_path: str | None = None,
    ) -> list[Document]:
        pass

    @cached(
        cache=Cache.REDIS,
        endpoint=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        serializer=PickleSerializer(),
        key_builder=get_cache_key,
        noself=True,
    )
    async def get_documents_cached(
        self,
        source_path: str | None = None,
    ) -> list[Document]:
        return await self.get_documents(source_path=source_path)

    async def load(
        self,
        source_path: str | None = None,
        cached_load: bool = False,
        pbar: tqdm | None = None,
    ) -> list[Document]:
        async with self.semaphore:
            documents = (
                await self.get_documents(source_path=source_path)
                if not cached_load
                else await self.get_documents_cached(source_path=source_path)
            )

            file_name = (
                Path(source_path).stem if source_path is not None else None
            )

            if pbar is not None:
                pbar.update(1)

            return [
                Document(
                    **doc.model_dump()
                    | {
                        "metadata": doc.metadata
                        | {
                            "document_index": idx,
                            "document_id": xxhash.xxh64(doc.text).hexdigest(),
                            "file_name": file_name,
                        }
                    }
                )
                for idx, doc in enumerate(documents, start=1)
            ]

    async def batch_load(
        self,
        source_paths: list[str],
        cached_load: bool = False,
    ) -> list[Document]:
        with tqdm(  # type: ignore
            total=len(source_paths),
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self.load(
                            source_path=source_path,
                            cached_load=cached_load,
                            pbar=pbar,
                        )
                    )
                    for source_path in source_paths
                ]

            return list(flatten((t.result() for t in tasks)))
