from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)

from rage.api.utils import get_filter, get_retriever
from rage.retriever import Retriever, RetrieverItem


class Filter(BaseModel):
    property: StrictStr = Field(
        description="Metadata property used to restrict retrieval results."
    )
    value: StrictStr | StrictInt | StrictFloat = Field(
        description="Exact value that the metadata property must match."
    )


class RetrieverInput(BaseModel):
    collection_names: list[StrictStr] = Field(
        description="Names of the collections to search."
    )
    query_text: StrictStr = Field(
        description="Natural-language or keyword query used for retrieval."
    )
    translate_query: bool = Field(
        default=True,
        description=(
            "Compatibility option requesting query translation; translation is not "
            "currently available in this package."
        ),
    )
    enable_reranker: StrictBool = Field(
        default=True,
        description=(
            "Compatibility option requesting result reranking; reranking is not "
            "currently available in this package."
        ),
    )
    enable_llm_response: StrictBool = Field(
        default=False,
        description=(
            "Compatibility option requesting a generated answer; answer generation "
            "is not currently available in this package."
        ),
    )
    wv_alpha: NonNegativeFloat = Field(
        default=0.7,
        le=1.0,
        description=(
            "Search mode selector: zero performs sparse search and any positive value "
            "performs hybrid search."
        ),
    )
    wv_min_similarity: StrictFloat = Field(
        default=0.4,
        description="Minimum score accepted from hybrid retrieval.",
    )
    retriever_limit: PositiveInt = Field(
        default=10,
        description="Maximum number of combined retrieval results to return.",
    )
    filters: list[Filter] = Field(
        default_factory=list,
        description="Exact-match metadata filters applied to every collection search.",
    )
    min_merging_chunks: PositiveInt = Field(
        default=2,
        description=(
            "Compatibility option for hierarchical merging; hierarchical retrieval "
            "is not currently available in this package."
        ),
    )


class RetrieverOutputItem(BaseModel):
    document_name: StrictStr = Field(
        description="Name of the source document containing the text chunk."
    )
    text: StrictStr = Field(description="Retrieved text chunk.")
    text_with_context: StrictStr | None = Field(
        default=None,
        description="Text chunk enriched with contextual metadata when available.",
    )
    document_metadata: dict = Field(
        default_factory=dict,
        description="Metadata stored with the source text chunk.",
    )
    node_id: StrictStr | None = Field(
        default=None,
        description="Hierarchical node identifier when available.",
    )
    parent_node_id: StrictStr | None = Field(
        default=None,
        description="Identifier of the hierarchical parent node when available.",
    )
    chunk_size: int = Field(
        default=0,
        description="Stored token count or size of the text chunk.",
    )
    chunk_id: int = Field(description="Sequential identifier of the text chunk.")
    keyword_score: NonNegativeFloat = Field(
        default=0.0,
        description="Keyword relevance score represented by the Qdrant result score.",
    )
    vector_score: NonNegativeFloat = Field(
        default=0.0,
        description="Vector relevance score represented by the Qdrant result score.",
    )
    relative_hybrid_score: NonNegativeFloat = Field(
        default=0.0,
        description="Combined score used to order the returned retrieval results.",
    )
    collection_metadata: dict = Field(
        default_factory=dict,
        description="Metadata identifying the collection that produced the result.",
    )
    child_node_ids: list[StrictStr] = Field(
        default_factory=list,
        description="Identifiers of hierarchical child nodes when available.",
    )


class RetrieverOutput(BaseModel):
    retriever_items: list[RetrieverOutputItem] = Field(
        default_factory=list,
        description="Text chunks returned by retrieval before optional post-processing.",
    )
    relevant_items: list[RetrieverOutputItem] = Field(
        default_factory=list,
        description="Relevant text chunks after optional post-processing.",
    )
    llm_response: StrictStr | None = Field(
        default=None,
        description="Generated answer when LLM response generation is available.",
    )
    error: StrictBool = Field(
        default=False,
        description="Whether retrieval failed because no requested collection exists.",
    )
    invalid_collections: list[StrictStr] = Field(
        default_factory=list,
        description="Requested collection names that do not exist.",
    )


def _retriever_item(
    item: RetrieverItem,
    collection_name: str,
    chunk_id: int,
) -> RetrieverOutputItem:
    metadata = item.metadata
    score = item.score or 0.0
    return RetrieverOutputItem(
        document_name=str(
            metadata.get("document_name", metadata.get("file_name", ""))
        ),
        text=item.text,
        text_with_context=metadata.get("text_with_context"),
        document_metadata=metadata,
        node_id=metadata.get("node_id"),
        parent_node_id=metadata.get("parent_node_id"),
        chunk_size=int(metadata.get("num_tokens", 0)),
        chunk_id=int(metadata.get("chunk_index", chunk_id)),
        keyword_score=score,
        vector_score=score,
        relative_hybrid_score=score,
        collection_metadata={"name": collection_name},
    )


retrieve_router = APIRouter()


@retrieve_router.post(
    "/rage/retriever/retrieve",
    tags=["retriever"],
    summary="Retrieve text chunks",
    description=(
        "Searches one or more Qdrant collections, combines their results by score, "
        "and reports collection names that could not be found."
    ),
)
async def retrieve(
    request: RetrieverInput,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RetrieverOutput:
    existence = [
        await retriever.qadrant_async_client.collection_exists(name)
        for name in request.collection_names
    ]
    valid_collections = [
        name
        for name, exists in zip(request.collection_names, existence, strict=True)
        if exists
    ]
    invalid_collections = [
        name
        for name, exists in zip(request.collection_names, existence, strict=True)
        if not exists
    ]
    if not valid_collections:
        return RetrieverOutput(error=True, invalid_collections=invalid_collections)

    search_filter = get_filter(request.filters)
    result_groups: list[tuple[str, list[RetrieverItem]]] = []
    for collection_name in valid_collections:
        if request.wv_alpha == 0.0:
            results = await retriever.sparse_search(
                collection_name,
                request.query_text,
                k=request.retriever_limit,
                search_filter=search_filter,
            )
        else:
            results = await retriever.hybrid_search(
                collection_name,
                request.query_text,
                k=request.retriever_limit,
                score_threshold=request.wv_min_similarity,
                search_filter=search_filter,
            )

        result_groups.append((collection_name, results))

    items = [
        _retriever_item(item, collection_name, chunk_id)
        for collection_name, results in result_groups
        for chunk_id, item in enumerate(results, start=1)
    ]
    items = sorted(
        items,
        key=lambda item: item.relative_hybrid_score,
        reverse=True,
    )[: request.retriever_limit]
    return RetrieverOutput(
        retriever_items=items,
        relevant_items=items,
        invalid_collections=invalid_collections,
    )
