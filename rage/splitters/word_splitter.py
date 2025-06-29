from tqdm import tqdm
from more_itertools import windowed, flatten

from common.logger import get_logger
from rage.meta.interfaces import TextSplitter, Document, TextChunk


logger = get_logger(__name__)


class WordSplitter(TextSplitter):
    def __init__(
        self,
        chunk_size: int = 128,
        chunk_overlap: int = 16,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _get_text_chunks(self, document: Document) -> list[TextChunk]:
        document_text = document.text
        document_metadata = document.metadata

        if document.is_table:
            return [
                TextChunk(
                    text=document_text,
                    metadata=document_metadata,
                    is_table=True,
                    num_tokens=self._get_num_tokens(text=document_text),
                )
            ]

        text_words = document_text.split()
        if len(text_words) <= self.chunk_size:
            return [
                TextChunk(
                    text=document_text,
                    metadata=document_metadata,
                    num_tokens=self._get_num_tokens(text=document_text),
                )
            ]

        step = self.chunk_size - self.chunk_overlap
        windows = list(
            windowed(
                text_words,
                n=self.chunk_size,
                step=step,
            )
        )

        # NOTE: Return the windows if the last window is complete.
        if None not in windows[-1]:
            text_chunks = (" ".join(words) for words in windows)
            return [
                TextChunk(
                    text=text,
                    metadata=document_metadata,
                    num_tokens=self._get_num_tokens(text=text),
                )
                for text in text_chunks
            ]

        # NOTE: If the last window is incomplete,
        # merge it with the second-to-last window.
        last_window = windows.pop()
        second_last_window = windows.pop()

        merged_window = second_last_window + tuple(
            last_window[self.chunk_overlap :]  # noqa
        )

        windows.append(merged_window)
        text_chunks = (
            " ".join(w for w in words if w is not None) for words in windows
        )

        return [
            TextChunk(
                text=text,
                metadata=document_metadata,
                num_tokens=self._get_num_tokens(text=text),
            )
            for text in text_chunks
        ]

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        text_chunks = map(self._get_text_chunks, tqdm(documents))
        return list(flatten(text_chunks))
