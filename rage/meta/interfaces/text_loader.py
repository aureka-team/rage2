import joblib
import asyncio

from tqdm import tqdm
from pathlib import Path
from more_itertools import flatten
from abc import ABC, abstractmethod
from pydantic import BaseModel, StrictStr, StrictBool, Field


class Document(BaseModel):
    text: StrictStr = Field(min_length=1)
    metadata: dict = {}
    is_table: StrictBool = False


class TextLoader(ABC):
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)

    @abstractmethod
    async def get_documents(
        self,
        source_path: str | None = None,
    ) -> list[Document]:
        pass

    async def load(
        self,
        source_path: str | None = None,
        pbar: tqdm | None = None,
    ) -> list[Document]:
        async with self.semaphore:
            documents = await self.get_documents(source_path=source_path)
            if pbar is not None:
                pbar.update(1)

            file_name = (
                Path(source_path).stem if source_path is not None else None
            )

            return [
                Document(
                    **doc.model_dump()
                    | {
                        "metadata": doc.metadata
                        | {
                            "document_index": idx,
                            "document_id": joblib.hash(doc.text),
                            "file_name": file_name,
                            "is_table": doc.is_table,
                        }
                    }
                )
                for idx, doc in enumerate(documents, start=1)
            ]

    async def batch_load(self, source_paths: list[str]) -> list[Document]:
        with tqdm(
            total=len(source_paths),
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self.load(
                            source_path=source_path,
                            pbar=pbar,
                        )
                    )
                    for source_path in source_paths
                ]

            return list(flatten((t.result() for t in tasks)))
