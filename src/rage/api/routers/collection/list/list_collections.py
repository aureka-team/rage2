from typing import Annotated

from fastapi import APIRouter, Depends

from rage.api.collection_metadata import list_resource_names
from rage.api.utils import get_retriever
from rage.retriever import Retriever

collection_list_router = APIRouter()


@collection_list_router.post(
    "/rage/collection/list",
    tags=["collection"],
    summary="List collections",
    description="Returns the names of the resources in the shared collection.",
)
async def list_collections(
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> list[str]:
    return await list_resource_names(retriever)
