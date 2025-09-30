import asyncio

from markitdown import MarkItDown

from common.cache import RedisCache
from common.logger import get_logger
from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


class DocxLoader(TextLoader):
    def __init__(
        self,
        cache: RedisCache | None = None,
    ):
        super().__init__(cache=cache)

    def _get_documents(self, source_path: str) -> list[Document]:
        md = MarkItDown()
        result = md.convert(source_path)

        return [Document(text=result.text_content)]

    async def get_documents(
        self,
        source_path: str | None = None,
    ) -> list[Document]:
        if source_path is None:
            return []

        return await asyncio.to_thread(
            self._get_documents,
            source_path=source_path,
        )
