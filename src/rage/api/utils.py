from functools import lru_cache
from typing import Protocol

from langchain_openai import OpenAIEmbeddings
from qdrant_client import models

from rage.api.config import api_config
from rage.retriever import Retriever


class FilterItem(Protocol):
    property: str
    value: str | int | float


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(
        dense_embeddings=OpenAIEmbeddings(
            model=api_config.emb_model,
            dimensions=api_config.emb_dimensions,
        )
    )


def get_filter(filters: list[FilterItem]) -> models.Filter | None:
    if not filters:
        return None

    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"metadata.{item.property}",
                match=models.MatchValue(value=item.value),
            )
            for item in filters
        ]
    )
