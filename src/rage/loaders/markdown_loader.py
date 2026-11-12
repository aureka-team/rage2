import asyncio
from functools import lru_cache
from threading import Lock

from markitdown import MarkItDown

from rage.meta.interfaces import Document, TextLoader

_conversion_lock = Lock()


@lru_cache(maxsize=1)
def _get_markitdown() -> MarkItDown:
    return MarkItDown()


class MarkdownLoader(TextLoader):
    def _get_documents(self, source_path: str) -> list[Document]:
        with _conversion_lock:
            markdown = _get_markitdown().convert(source=source_path)

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
