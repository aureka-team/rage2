from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastembed.sparse.bm25 import Bm25
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)

from rage.api.collection_metadata import get_resource_metadata
from rage.api.shared_index import CONTENT_COLLECTION, generation_filter
from rage.api.utils import get_filter, get_retriever
from rage.llm_agents import TextChunk, reranker_agent, retrieval_assistant_agent
from rage.retriever import Retriever, RetrieverItem
from rage.utils.keyword_matches import keyword_match_ranges


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
    enable_llm_response: StrictBool = Field(
        default=False,
        description="Whether to answer the query using the relevant retrieved items.",
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
    match_ranges: list[tuple[NonNegativeInt, NonNegativeInt]] | None = Field(
        default=None,
        description="Half-open UTF-16 offsets in text for BM25 matches.",
    )
    document_metadata: dict = Field(
        default_factory=dict,
        description="Metadata stored with the source text chunk.",
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
    llm_response: StrictStr | None = Field(
        default=None,
        description="Answer generated from the relevant items when enabled.",
    )
    error: StrictBool = Field(
        default=False,
        description="Whether any requested collection does not exist.",
    )
    invalid_collections: list[StrictStr] = Field(
        default_factory=list,
        description="All nonexistent collection names, using the exact request identifiers in request order.",
    )


def _retriever_item(
    item: RetrieverItem,
    collection_name: str,
) -> RetrieverOutputItem:
    metadata = item.metadata
    score = item.score or 0.0
    return RetrieverOutputItem(
        document_name=str(metadata.get("document_name", metadata.get("file_name", ""))),
        text=item.text,
        document_metadata=metadata,
        score=score,
        collection_metadata={"name": collection_name},
    )


async def _rerank_items(
    items: list[RetrieverOutputItem],
    query_text: str,
) -> list[RetrieverOutputItem]:
    indexed_items = dict(enumerate(items, start=1))
    text_chunks = [
        TextChunk(chunk_id=chunk_id, text=item.text)
        for chunk_id, item in indexed_items.items()
    ]
    reranker_result = await reranker_agent.run(
        (
            f"**Query**: {query_text}\n\n"
            f"**Text Chunks**: {[chunk.model_dump() for chunk in text_chunks]}"
        ),
    )

    return [
        indexed_items[chunk_id]
        for chunk_id in reranker_result.output.relevant_chunk_ids
        if chunk_id in indexed_items
    ]


async def _generate_llm_response(
    items: list[RetrieverOutputItem],
    query_text: str,
) -> str | None:
    retrieval_assistant_result = await retrieval_assistant_agent.run(
        (
            f"**Query**: {query_text}\n\n"
            f"**Relevant Text Chunks**: "
            f"{[{'text': item.text} for item in items]}"
        ),
    )

    return retrieval_assistant_result.output.response


retrieve_router = APIRouter()


@retrieve_router.post(
    "/rage/retriever/retrieve",
    tags=["retriever"],
    summary="Retrieve text chunks",
    description=(
        "Searches the requested resources in the shared Qdrant collection, "
        "and reports every nonexistent collection in request order. Missing "
        "collections set error=true; existing collections are still searched. "
        "An empty result with error=false means all collections exist."
    ),
)
async def retrieve(
    retrieve_input: RetrieverInput,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RetrieverOutput:
    resources = await get_resource_metadata(retriever, retrieve_input.collection_names)
    invalid_collections = [
        name for name in retrieve_input.collection_names if name not in resources
    ]
    if not resources:
        return RetrieverOutput(
            error=True,
            invalid_collections=invalid_collections,
        )

    search_filter = generation_filter(
        [resource["generation"] for resource in resources.values()],
        get_filter(retrieve_input.filters),
    )
    query = retrieve_input.query_text
    limit = retrieve_input.retriever_limit
    match retrieve_input.search_mode:
        case "dense":
            results = await retriever.dense_search(
                CONTENT_COLLECTION,
                query,
                k=limit,
                score_threshold=retrieve_input.min_similarity,
                search_filter=search_filter,
            )
        case "sparse":
            results = await retriever.sparse_search(
                CONTENT_COLLECTION, query, k=limit, search_filter=search_filter
            )
        case "hybrid":
            results = await retriever.hybrid_search(
                CONTENT_COLLECTION,
                query,
                k=limit,
                score_threshold=retrieve_input.min_similarity,
                search_filter=search_filter,
            )

    items = [
        _retriever_item(item, item.metadata["rage_resource_id"]) for item in results
    ]
    if retrieve_input.search_mode == "sparse" and items:
        model = retriever.sparse_embeddings._model.model
        if isinstance(model, Bm25):
            query_vector = next(model.query_embed(retrieve_input.query_text))
            query_token_ids = set(query_vector.indices.tolist())
            for item in items:
                item.match_ranges = keyword_match_ranges(
                    item.text, model, query_token_ids
                )

    relevant_items = items
    if retrieve_input.enable_reranker or retrieve_input.enable_llm_response:
        relevant_items = await _rerank_items(items, retrieve_input.query_text)

    llm_response = None
    if retrieve_input.enable_llm_response and relevant_items:
        llm_response = await _generate_llm_response(
            relevant_items,
            retrieve_input.query_text,
        )

    return RetrieverOutput(
        retriever_items=items,
        relevant_items=relevant_items,
        llm_response=llm_response,
        error=bool(invalid_collections),
        invalid_collections=invalid_collections,
    )
