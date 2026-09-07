from rage.meta.interfaces import Document, TextChunk, TextSplitter


class DocumentSplitter(TextSplitter):
    def __init__(self):
        super().__init__()

    def _split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        return [
            TextChunk(
                text=doc.text,
                metadata=doc.metadata,
                num_tokens=self._get_num_tokens(text=doc.text),
            )
            for doc in documents
        ]
