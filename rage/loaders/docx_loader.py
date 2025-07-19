import asyncio

from bs4 import BeautifulSoup
from unstructured.partition.docx import partition_docx

from common.logger import get_logger
from rage.meta.interfaces import TextLoader, Document


logger = get_logger(__name__)


class DocxLoader(TextLoader):
    def __init__(self):
        super().__init__()

    def _get_table_document(self, table_element: dict) -> Document:
        table_html = table_element["metadata"]["text_as_html"]
        soup = BeautifulSoup(table_html, "html.parser")

        return Document(
            text=soup.prettify(),
            is_table=True,
        )

    def _get_documents(self, source_path: str) -> list[Document]:
        text_elements = partition_docx(
            filename=source_path,
            include_page_breaks=False,
        )

        text_elements = [te.to_dict() for te in text_elements]
        text_document = Document(
            text=" ".join(te["text"] for te in text_elements).strip()
        )

        table_documents = [
            self._get_table_document(table_element=te)
            for te in text_elements
            if te["type"] == "Table"
        ]

        documents = [text_document] + table_documents
        return [doc for doc in documents if doc.text]

    async def get_documents(self, source_path: str) -> list[Document]:
        return await asyncio.to_thread(
            self._get_documents,
            source_path=source_path,
        )
