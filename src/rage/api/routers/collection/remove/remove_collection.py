from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StrictBool, StrictStr

from rage.api.utils import get_retriever
from rage.retriever import Retriever


class RemoveCollectionInput(BaseModel):
    name: StrictStr = Field(
        description="Name of the collection to delete."
    )


class RemoveCollectionOutput(BaseModel):
    removed: StrictBool = Field(
        description="Whether the collection existed and was deleted."
    )


collection_remove_router = APIRouter()


@collection_remove_router.post(
    "/rage/collection/remove",
    tags=["collection"],
    summary="Remove a collection",
    description=(
        "Deletes the named Qdrant collection and reports whether it existed."
    ),
)
async def remove_collection(
    request: RemoveCollectionInput,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RemoveCollectionOutput:
    exists = await retriever.qadrant_async_client.collection_exists(request.name)
    if exists:
        await retriever.qadrant_async_client.delete_collection(request.name)

    return RemoveCollectionOutput(removed=exists)
