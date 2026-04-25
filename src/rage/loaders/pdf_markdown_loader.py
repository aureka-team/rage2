import asyncio
import pymupdf4llm

from rich.console import Console
from rage.meta.interfaces import TextLoader, Document


console = Console()


class PDFMarkdownLoader(TextLoader):
    def __init__(self):
        super().__init__()

    def _get_documents(self, source_path: str) -> list[Document]:
        md_text = pymupdf4llm.to_markdown(
            source_path,
            use_ocr=False,
            ignore_images=True,
            ignore_graphics=True,
            show_progress=True,
        )

        if not len(md_text):
            console.log(
                f"[bold yellow]WARNING:[/] no text in file: {source_path}"
            )

            return []

        return [Document(text=md_text)]  # type: ignore

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
