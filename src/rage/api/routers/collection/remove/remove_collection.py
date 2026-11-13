from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StrictBool, StrictStr

from rage.api.app_index import AppIndexDependency
from rage.api.collection_metadata import (
    get_collection_metadata,
    remove_collection_metadata,
)
from rage.api.shared_index import delete_resource_chunks, resource_write_lock
from rage.api.utils import get_retriever
from rage.retriever import Retriever


class RemoveCollectionInput(BaseModel):
    name: StrictStr = Field(description="Name of the collection to delete.")


class RemoveCollectionOutput(BaseModel):
    removed: StrictBool = Field(
        description="Whether the collection existed and was deleted."
    )


collection_remove_router = APIRouter()


@collection_remove_router.post(
    "/rage/collection/remove",
    tags=["collection"],
    summary="Remove a collection",
    description=("Removes the named resource from the shared collection."),
)
async def remove_collection(
    request: RemoveCollectionInput,
    index: AppIndexDependency,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RemoveCollectionOutput:
    async with resource_write_lock(index.content, request.name):
        exists = (
            await get_collection_metadata(retriever, index.registry, request.name)
            is not None
        )
        await remove_collection_metadata(retriever, index.registry, request.name)
        await delete_resource_chunks(retriever, index.content, request.name)
    return RemoveCollectionOutput(removed=exists)
