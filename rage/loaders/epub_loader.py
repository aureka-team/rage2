import asyncio

from unstructured.partition.epub import partition_epub

from common.logger import get_logger
from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


BANNED_TYPES = {"Image"}


class EpubLoader(TextLoader):
    def __init__(self, banned_types: set[str] = BANNED_TYPES):
        super().__init__()
        self.banned_types = banned_types

    def _get_documents(self, source_path) -> list[Document]:
        text_elements = partition_epub(filename=source_path)
        text_elements = [te.to_dict() for te in text_elements]

        documents = (
            Document(
                text=te["text"].strip(),
                metadata={
                    "type": te["type"],
                },
            )
            for te in text_elements
            if te["type"] not in self.banned_types
        )

        return [doc for doc in documents if doc.text]

    async def get_documents(self, source_path) -> list[Document]:
        return await asyncio.to_thread(
            self._get_documents,
            source_path=source_path,
        )
