import asyncio
import pymupdf4llm

from common.logger import get_logger
from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


class PDFMarkdownLoaeder(TextLoader):
    def __init__(self):
        super().__init__()

    def _get_documents(self, source_path: str) -> list[Document]:
        md_text = pymupdf4llm.to_markdown(source_path)
        return [Document(text=md_text)]

    async def get_documents(self, source_path: str) -> list[Document]:
        return await asyncio.to_thread(
            self._get_documents,
            source_path=source_path,
        )
