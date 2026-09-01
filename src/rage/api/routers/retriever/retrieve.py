from typing import Annotated, Literal

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
from rage.llm_agents import Reranker, RerankerDeps, TextChunk
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
    search_mode: Literal["dense", "hybrid", "sparse"] = Field(
        default="hybrid",
        description="Search type to perform.",
    )
    enable_reranker: StrictBool = Field(
        default=False,
        description="Whether to select and reorder retrieved items with the reranker.",
    )
    min_similarity: StrictFloat = Field(
        default=0.3,
        description="Minimum score accepted from dense and hybrid retrieval.",
    )
    retriever_limit: PositiveInt = Field(
        default=10,
        description="Maximum number of combined retrieval results to return.",
    )
    filters: list[Filter] = Field(
        default_factory=list,
        description="Exact-match metadata filters applied to every collection search.",
    )


class RetrieverOutputItem(BaseModel):
    document_name: StrictStr = Field(
        description="Name of the source document containing the text chunk."
    )
    text: StrictStr = Field(description="Retrieved text chunk.")
    document_metadata: dict = Field(
        default_factory=dict,
        description="Metadata stored with the source text chunk.",
    )
    chunk_size: int = Field(
        default=0,
        description="Stored token count or size of the text chunk.",
    )
    chunk_id: int = Field(
        description="Sequential identifier of the text chunk."
    )
    keyword_score: NonNegativeFloat = Field(
        default=0.0,
        description="Keyword relevance score represented by the Qdrant result score.",
    )
    vector_score: NonNegativeFloat = Field(
        default=0.0,
        description="Vector relevance score represented by the Qdrant result score.",
    )
    score: NonNegativeFloat = Field(
        default=0.0,
        description="Retrieval score used to order the returned results.",
    )
    collection_metadata: dict = Field(
        default_factory=dict,
        description="Metadata identifying the collection that produced the result.",
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
        document_metadata=metadata,
        chunk_size=int(metadata.get("num_tokens", 0)),
        chunk_id=int(metadata.get("chunk_index", chunk_id)),
        keyword_score=score,
        vector_score=score,
        score=score,
        collection_metadata={"name": collection_name},
    )


async def _rerank_items(
    items: list[RetrieverOutputItem],
    query_text: str,
) -> list[RetrieverOutputItem]:
    indexed_items = dict(enumerate(items, start=1))
    reranker_output = await Reranker().generate(
        user_prompt=query_text,
        agent_deps=RerankerDeps(
            text_chunks=[
                TextChunk(chunk_id=chunk_id, text=item.text)
                for chunk_id, item in indexed_items.items()
            ],
            query_text=query_text,
        ),
    )

    return [
        indexed_items[chunk_id]
        for chunk_id in reranker_output.relevant_chunk_ids
        if chunk_id in indexed_items
    ]


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
    retrieve_input: RetrieverInput,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RetrieverOutput:
    existence = [
        await retriever.qadrant_async_client.collection_exists(name)
        for name in retrieve_input.collection_names
    ]
    valid_collections = [
        name
        for name, exists in zip(
            retrieve_input.collection_names, existence, strict=True
        )
        if exists
    ]
    invalid_collections = [
        name
        for name, exists in zip(
            retrieve_input.collection_names, existence, strict=True
        )
        if not exists
    ]
    if not valid_collections:
        return RetrieverOutput(
            error=True,
            invalid_collections=invalid_collections,
        )

    search_filter = get_filter(retrieve_input.filters)
    result_groups: list[tuple[str, list[RetrieverItem]]] = []
    for collection_name in valid_collections:
        match retrieve_input.search_mode:
            case "dense":
                results = await retriever.dense_search(
                    collection_name,
                    retrieve_input.query_text,
                    k=retrieve_input.retriever_limit,
                    score_threshold=retrieve_input.min_similarity,
                    search_filter=search_filter,
                )

            case "sparse":
                results = await retriever.sparse_search(
                    collection_name,
                    retrieve_input.query_text,
                    k=retrieve_input.retriever_limit,
                    search_filter=search_filter,
                )

            case "hybrid":
                results = await retriever.hybrid_search(
                    collection_name,
                    retrieve_input.query_text,
                    k=retrieve_input.retriever_limit,
                    score_threshold=retrieve_input.min_similarity,
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
        key=lambda item: item.score,
        reverse=True,
    )[: retrieve_input.retriever_limit]
    relevant_items = items
    if retrieve_input.enable_reranker:
        relevant_items = await _rerank_items(items, retrieve_input.query_text)

    return RetrieverOutput(
        retriever_items=items,
        relevant_items=relevant_items,
        invalid_collections=invalid_collections,
    )
