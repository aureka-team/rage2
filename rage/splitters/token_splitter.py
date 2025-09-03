from more_itertools import flatten

from langchain_text_splitters import TokenTextSplitter
from rage.meta.interfaces import TextSplitter, Document, TextChunk


class TokenSplitter(TextSplitter):
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
    ):
        super().__init__()

        self.splitter = TokenTextSplitter(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def get_text_chunks(self, document: Document) -> list[TextChunk]:
        text_chunks = self.splitter.split_text(text=document.text)
        text_chunks = (tc.strip() for tc in text_chunks)

        return [
            TextChunk(
                text=text,
                metadata=document.metadata,
                num_tokens=self._get_num_tokens(text=text),
            )
            for text in text_chunks
        ]

    def _split_documents(self, documents: list[Document]) -> list[TextChunk]:
        text_chunks = map(self.get_text_chunks, documents)
        return list(flatten(text_chunks))
