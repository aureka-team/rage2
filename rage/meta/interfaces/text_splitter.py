import joblib
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
        text_chunks_ = self._split_documents(documents=documents)
        text_chunks_ = [
            TextChunk(
                text=tc.text,
                metadata=tc.metadata
                | {
                    "chunk_id": joblib.hash(tc.text),
                    "chunk_idx": idx,
                },
                is_table=tc.is_table,
                num_tokens=tc.num_tokens,
            )
            for idx, tc in enumerate(text_chunks_, start=1)
        ]

        text_chunks = []
        for idx, tc in enumerate(text_chunks_):
            previous_chunk_id = (
                None if idx == 0 else text_chunks_[idx - 1].metadata["chunk_id"]
            )

            next_chunk_id = (
                None
                if idx == len(text_chunks_) - 1
                else text_chunks_[idx + 1].metadata["chunk_id"]
            )

            text_chunks.append(
                TextChunk(
                    text=tc.text,
                    metadata=tc.metadata
                    | {
                        "previous_chunk_id": previous_chunk_id,
                        "next_chunk_id": next_chunk_id,
                    },
                    is_table=tc.is_table,
                    num_tokens=tc.num_tokens,
                )
            )

        return text_chunks
