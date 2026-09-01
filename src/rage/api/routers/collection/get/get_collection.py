from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StrictStr

from rage.api.collection_metadata import get_collection_metadata
from rage.api.utils import get_retriever
from rage.retriever import Retriever


class GetCollectionInput(BaseModel):
    collection_name: StrictStr = Field(
        description="Name of the collection whose metadata should be returned."
    )


class GetCollectionOutput(BaseModel):
    collection_name: StrictStr | None = Field(
        default=None,
        description="Name of the collection, or null when it does not exist.",
    )
    collection_metadata: dict | None = Field(
        default=None,
        description="Stored language and source-document metadata for the collection.",
    )


collection_get_router = APIRouter()


@collection_get_router.post(
    "/rage/collection/get",
    tags=["collection"],
    summary="Get collection metadata",
    description=(
        "Returns metadata stored in the collection metadata registry. "
        "An empty response is returned when the collection does not exist."
    ),
)
async def get_collection(
    request: GetCollectionInput,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> GetCollectionOutput:
    exists = await retriever.qadrant_async_client.collection_exists(
        request.collection_name
    )
    if not exists:
        return GetCollectionOutput()

    return GetCollectionOutput(
        collection_name=request.collection_name,
        collection_metadata=await get_collection_metadata(
            retriever,
            request.collection_name,
        ),
    )
