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
        chunk_size: int = 128,
        chunk_overlap: int = 8,
        tt_encoder_name: str = "gpt-4o",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tt_encoder = tiktoken.encoding_for_model(tt_encoder_name)

    def _get_num_tokens(self, text: str) -> int:
        return len(self.tt_encoder.encode(text))

    @abstractmethod
    def split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        pass
