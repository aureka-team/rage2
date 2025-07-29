import asyncio

from markitdown import MarkItDown

from common.logger import get_logger
from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


class MarkdownLoader(TextLoader):
    def __init__(self):
        super().__init__()
        self.markitdown = MarkItDown()

    def _get_documents(self, source_path: str) -> list[Document]:
        markdown = self.markitdown.convert(source=source_path)
        return [Document(text=markdown.markdown)]

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
