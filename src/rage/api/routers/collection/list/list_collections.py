from typing import Annotated

from fastapi import APIRouter, Depends

from rage.api.utils import get_retriever
from rage.retriever import Retriever

collection_list_router = APIRouter()


@collection_list_router.post(
    "/rage/collection/list",
    tags=["collection"],
    summary="List collections",
    description="Returns the names of all collections currently stored in Qdrant.",
)
async def list_collections(
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> list[str]:
    response = await retriever.qadrant_async_client.get_collections()
    return [collection.name for collection in response.collections]
