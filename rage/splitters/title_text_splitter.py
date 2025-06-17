from more_itertools import split_before

from common.logger import get_logger
from rage.meta.interfaces import TextSplitter, Document, TextChunk


logger = get_logger(__name__)


class TitleSplitter(TextSplitter):
    def __init__(self, title_tag: str = "title"):
        super().__init__()
        self.title_tag = title_tag

    def _get_text_chunk(self, title_group: list[Document]) -> TextChunk:
        text = " ".join(doc.text for doc in title_group)
        return TextChunk(
            text=text,
            metadata={"title": title_group[0].text},
            num_tokens=self._get_num_tokens(text=text),
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        title_groups = split_before(
            documents,
            lambda x: x.metadata["type"].lower() == self.title_tag,
        )

        return [
            self._get_text_chunk(title_group=title_group)
            for title_group in title_groups
        ]
