from tqdm import tqdm
from pypdf import PdfReader

from common.cache import RedisCache
from common.logger import get_logger

from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


class PDFLoaeder(TextLoader):
    def __init__(
        self,
        cache: RedisCache | None = None,
        disable_progress: bool = False,
    ):
        super().__init__(cache=cache)
        self.disable_progress = disable_progress

    def _get_documents(self, source_path) -> list[Document]:
        reader = PdfReader(source_path)
        documents = (
            Document(
                text=page.extract_text().strip(),
                metadata={
                    "page_number": page.page_number,
                },
            )
            for page in tqdm(
                iterable=reader.pages,
                disable=self.disable_progress,
            )
        )

        return [doc for doc in documents if doc.text]
