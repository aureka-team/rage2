import os
import json
import requests

from typing import Literal
from langchain_core.embeddings import Embeddings


IONOS_TOKEN = os.getenv("IONOS_TOKEN")
EMBEDDING_DIMENSIONS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
}


class IonosEmbeddings(Embeddings):
    def __init__(
        self,
        model: Literal[
            "BAAI/bge-m3",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ] = "BAAI/bge-m3",
        endpoint: str = "https://openai.inference.de-txl.ionos.com/v1/embeddings",
    ):
        super().__init__()

        self.model = model
        self.dimensions = EMBEDDING_DIMENSIONS[model]

        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {IONOS_TOKEN}",
            "Content-Type": "application/json",
        }

    def get_embeddings_data_items_(self, texts: list[str]) -> list[dict]:
        body = {
            "model": self.model,
            "input": texts,
        }

        result = requests.post(
            self.endpoint,
            json=body,
            headers=self.headers,
        )

        status_code = result.status_code
        assert status_code == 200, f"status_code: {status_code}"
        return json.loads(result.content.decode())["data"]

    def embed_query(self, text: str) -> list[float]:
        data_items = self.get_embeddings_data_items_(texts=[text])
        return data_items[0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        data_items = self.get_embeddings_data_items_(texts=texts)
        return [data_item["embedding"] for data_item in data_items]
