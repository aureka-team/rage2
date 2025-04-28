import json

from more_itertools import partition
from unstructured.partition.docx import partition_docx

from rage.meta.interfaces import TextLoader, Document
from rage.llm_agents import TableExtractor, TableExtractorInput

from common.cache import RedisCache
from common.logger import get_logger


logger = get_logger(__name__)


class DocxLoader(TextLoader):
    def __init__(self, cache: RedisCache | None = None):
        super().__init__(cache=cache)

    def _get_document(self, text_element: dict) -> Document:
        return Document(
            text=text_element["text"],
            metadata=text_element["metadata"],
        )

    async def _get_table_documents(
        self, table_elements: list[dict]
    ) -> list[Document]:
        table_extractor_inputs = [
            TableExtractorInput(html_table_text=te["metadata"]["text_as_html"])
            for te in table_elements
        ]

        table_extractor = TableExtractor()
        table_extractor_outputs = await table_extractor.batch_generate(
            agent_inputs=table_extractor_inputs
        )

        return [
            Document(
                text=te["text"],
                metadata=te["metadata"],
                json_table=json.loads(teo.json_table),
            )
            for te, teo in zip(table_elements, table_extractor_outputs)
        ]

    async def _load(self, source_path) -> list[Document]:
        text_elements = partition_docx(
            filename=source_path,
            include_page_breaks=False,
        )

        text_elements = (te.to_dict() for te in text_elements)
        text_elements, table_elements = partition(
            lambda x: x["type"] == "Table", text_elements
        )

        documents = [
            Document(
                text=te["text"],
                metadata=te["metadata"],
            )
            for te in text_elements
        ]

        table_elements = list(table_elements)
        table_documents = await self._get_table_documents(
            table_elements=table_elements
        )

        return table_documents
        # return documents + table_documents
