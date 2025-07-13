import tiktoken

from abc import ABC, abstractmethod
from pydantic import NonNegativeInt

from common.logger import get_logger

from .text_loader import Document


logger = get_logger(__name__)


class TextChunk(Document):
    num_tokens: NonNegativeInt


class TextSplitter(ABC):
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        tt_encoder_name: str = "gpt-4o",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tt_encoder = tiktoken.encoding_for_model(tt_encoder_name)

    def _get_num_tokens(self, text: str) -> int:
        return len(self.tt_encoder.encode(text))

    @abstractmethod
    def _split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        pass

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        text_chunks = self._split_documents(documents=documents)
        return [
            TextChunk(
                text=tc.text,
                metadata=tc.metadata | {"chunk_id": idx},
                is_table=tc.is_table,
                num_tokens=tc.num_tokens,
            )
            for idx, tc in enumerate(text_chunks, start=1)
        ]
