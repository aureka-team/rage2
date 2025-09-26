# import joblib
import xxhash
import tiktoken

from abc import ABC, abstractmethod
from pydantic import NonNegativeInt

from common.logger import get_logger

from .text_loader import Document


logger = get_logger(__name__)


class TextChunk(Document):
    num_tokens: NonNegativeInt


class TextSplitter(ABC):
    def __init__(self, tt_encoder_name: str = "gpt-4o"):
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
                    # "chunk_id": joblib.hash(tc.text),
                    "chunk_id": xxhash.xxh64(tc.text).hexdigest(),
                    "chunk_index": idx,
                },
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
                    num_tokens=tc.num_tokens,
                )
            )

        return text_chunks
