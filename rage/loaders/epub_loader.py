import asyncio

from tqdm import tqdm
from unstructured.partition.epub import partition_epub

from rage.meta.interfaces import TextLoader, Document

from common.cache import RedisCache
from common.logger import get_logger


logger = get_logger(__name__)


BANNED_TYPES = {"Image"}


class EpubLoader(TextLoader):
    def __init__(
        self,
        banned_types: set[str] = BANNED_TYPES,
        cache: RedisCache | None = None,
    ):
        super().__init__(cache=cache)
        self.banned_types = banned_types

    def _get_documents(self, source_path) -> list[Document]:
        text_elements = partition_epub(filename=source_path)
        text_elements = [te.to_dict() for te in text_elements]

        return [
            Document(
                text=te["text"],
                metadata={
                    "type": te["type"],
                },
            )
            for te in text_elements
            if te["type"] not in self.banned_types
        ]

    async def _load(
        self,
        source_path,
        pb: tqdm | None = None,
    ) -> list[Document]:
        documents = await asyncio.to_thread(
            self._get_documents,
            source_path=source_path,
        )

        if pb is not None:
            pb.update(1)

        return documents
