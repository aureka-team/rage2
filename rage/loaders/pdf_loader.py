from tqdm import tqdm
from pathlib import Path
from pypdf import PdfReader

from common.cache import RedisCache
from common.logger import get_logger

from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


class PDFLoaeder(TextLoader):
    def __init__(self, cache: RedisCache | None = None):
        super().__init__(cache=cache)

    def _get_documents(self, source_path) -> list[Document]:
        book_name = Path(source_path).stem
        reader = PdfReader(source_path)
        return [
            Document(
                text=page.extract_text(),
                metadata={
                    "title": book_name,
                    "page_number": page.page_number,
                },
            )
            for page in tqdm(reader.pages)
        ]
