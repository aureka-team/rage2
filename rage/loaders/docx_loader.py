from rage.meta.interfaces import TextLoader, Document
from unstructured.partition.docx import partition_docx

from common.cache import RedisCache
from common.logger import get_logger


logger = get_logger(__name__)


class DocxLoader(TextLoader):
    def __init__(self, cache: RedisCache | None = None):
        super().__init__(cache=cache)

    def _load(self, source_path) -> list[Document]:
        text_elements = (
            elem.to_dict() for elem in partition_docx(filename=source_path)
        )

        return [Document(text=" ".join(te["text"] for te in text_elements))]
