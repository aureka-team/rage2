from common.logger import get_logger
from rage.meta.interfaces import TextSplitter, Document, TextChunk


logger = get_logger(__name__)


class DocumentSplitter(TextSplitter):
    def __init__(self):
        super().__init__()

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        return [
            TextChunk(
                text=doc.text,
                metadata=doc.metadata,
                chunk_id=idx,
                num_tokens=self._get_num_tokens(text=doc.text),
            )
            for idx, doc in enumerate(documents, start=1)
        ]
