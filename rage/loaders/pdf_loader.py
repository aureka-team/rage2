import asyncio

from tqdm import tqdm
from pypdf.errors import PdfReadError
from pypdf import PdfReader, PageObject

from rage.meta.interfaces import TextLoader, Document

from common.cache import RedisCache
from common.logger import get_logger

from rage.utils.text import fix_punctuation


logger = get_logger(__name__)


class PDFLoaeder(TextLoader):
    def __init__(self, cache: RedisCache | None = None):
        super().__init__(cache=cache)

    def _clean_text(self, text: str) -> str:
        # NOTE: Remove the soft hyphen used to split words
        text = text.replace("\xad", "")
        text = " ".join(text.split())

        return fix_punctuation(text=text)

    # FIXME: https://github.com/py-pdf/pypdf/issues/2866
    def _extract_page_text(self, page: PageObject) -> dict | None:
        try:
            return {
                "page_number": page.page_number,
                "text": self._clean_text(text=page.extract_text()),
            }

        except PdfReadError:
            logger.error(f"ERROR extracting page {page.page_number}")

    # FIXME: https://github.com/py-pdf/pypdf/issues/2866
    def _extract_page_texts(self, pages: list[PageObject]) -> list[dict]:
        page_texts = map(
            self._extract_page_text,
            tqdm(
                pages,
                ascii=" ##",
                colour="#808080",
                desc="parsing pdf",
            ),
        )

        return [page_text for page_text in page_texts if page_text is not None]

    def get_page_documents(
        self,
        pages: list[PageObject],
    ) -> list[Document]:
        page_texts = self._extract_page_texts(pages=pages)
        page_documents = (
            Document(
                text=page_text["text"],
                metadata={
                    "page_number": page_text["page_number"],
                },
            )
            for page_text in page_texts
        )

        return [document for document in page_documents if document.text]

    def _get_documents(self, source_path) -> list[Document]:
        reader = PdfReader(source_path)
        pages = reader.pages
        logger.info(f"pages => {len(pages)}")

        return self.get_page_documents(pages=pages)

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
