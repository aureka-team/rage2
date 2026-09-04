from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from qdrant_client import models

from rage.api.utils import get_filter, get_retriever
from rage.retriever import Retriever


class Filter(BaseModel):
    property: StrictStr = Field(
        description="Metadata property used to filter collection objects."
    )
    value: StrictStr | StrictInt | StrictFloat = Field(
        description="Exact value that the metadata property must match."
    )


class Sort(BaseModel):
    property: StrictStr = Field(
        description="Metadata property used to order collection objects."
    )
    ascending: StrictBool = Field(
        description="Sort ascending when true and descending when false."
    )


class GetCollectionObjectsInput(BaseModel):
    collection_name: StrictStr = Field(
        description="Name of the collection containing the requested objects."
    )
    properties: list[StrictStr] = Field(
        description="Payload properties to include in each returned object."
    )
    filters: list[Filter] = Field(
        description="Exact-match metadata filters applied to the objects."
    )
    sort: Sort | None = Field(
        default=None,
        description="Optional metadata ordering applied before limiting results.",
    )
    limit: PositiveInt | None = Field(
        default=20,
        description="Maximum number of objects to return; null uses the API default.",
    )


class CollectionObject(BaseModel):
    name: StrictStr = Field(
        description="Name of the collection containing the object."
    )
    uuid: StrictStr = Field(
        description="Qdrant point identifier represented as a string."
    )
    properties: dict = Field(
        description="Requested payload properties found on the object."
    )


def _record_properties(payload: dict[str, Any], properties: list[str]) -> dict:
    values = payload.get("metadata", {}) | {"text": payload.get("page_content")}
    return {name: values[name] for name in properties if name in values}


collection_get_objects_router = APIRouter()


@collection_get_objects_router.post(
    "/rage/collection/get_objects",
    tags=["collection"],
    summary="Get collection objects",
    description=(
        "Scrolls through collection objects with optional exact-match filters, "
        "metadata sorting, property selection, and a result limit."
    ),
)
async def get_collection_objects(
    request: GetCollectionObjectsInput,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> list[CollectionObject]:
    exists = await retriever.qadrant_async_client.collection_exists(
        request.collection_name
    )
    if not exists:
        return []

    order_by = None
    if request.sort is not None:
        order_by = models.OrderBy(
            key=f"metadata.{request.sort.property}",
            direction="asc" if request.sort.ascending else "desc",
        )

    records = await retriever.scroll(
        collection_name=request.collection_name,
        limit=request.limit or 10,
        scroll_filter=get_filter(request.filters),
        order_by=order_by,
    )
    return [
        CollectionObject(
            name=request.collection_name,
            uuid=str(record.id),
            properties=_record_properties(
                record.payload or {}, request.properties
            ),
        )
        for record in records
    ]
