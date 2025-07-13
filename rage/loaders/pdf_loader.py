from tqdm import tqdm
from pypdf import PdfReader

from common.cache import RedisCache
from common.logger import get_logger

from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


class PDFLoaeder(TextLoader):
    def __init__(self, cache: RedisCache | None = None):
        super().__init__(cache=cache)

    async def _load(
        self,
        source_path,
    ) -> list[Document]:
        reader = PdfReader(source_path)
        return [
            Document(
                text=page.extract_text(),
                metadata={"page_number": page.page_number},
            )
            for page in tqdm(reader.pages)
        ]
