"""Storage conventions for resources indexed through the RAGE HTTP API."""

from asyncio import Lock
from collections.abc import Sequence

from qdrant_client import models

from rage.retriever import Retriever

RESOURCE_FIELD = "metadata.rage_resource_id"
GENERATION_FIELD = "metadata.rage_generation"

# The API runs with one worker. Serialize writes to a resource, including
# deletion, without making the backfill wait for unrelated resources.
_write_locks = tuple(Lock() for _ in range(64))
_setup_lock = Lock()


def resource_write_lock(collection: str, name: str) -> Lock:
    return _write_locks[hash((collection, name)) % len(_write_locks)]


def generation_filter(
    generations: Sequence[str], filters: models.Filter | None = None
) -> models.Filter:
    scope = models.FieldCondition(
        key=GENERATION_FIELD, match=models.MatchAny(any=list(generations))
    )
    return models.Filter(must=[scope, filters] if filters else [scope])


async def ensure_content_collection(retriever: Retriever, collection: str) -> None:
    async with _setup_lock:
        await retriever.create_collection(collection)
        await retriever.ensure_payload_indexes(collection)
        await retriever.create_payload_index(collection, GENERATION_FIELD)
        await retriever.create_payload_index(collection, RESOURCE_FIELD)


async def delete_resource_chunks(
    retriever: Retriever, collection: str, name: str, keep_generation: str | None = None
) -> None:
    if not await retriever.qadrant_async_client.collection_exists(collection):
        return
    await retriever.qadrant_async_client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=RESOURCE_FIELD,
                        match=models.MatchValue(value=name),
                    )
                ],
                must_not=(
                    [generation_filter([keep_generation])] if keep_generation else None
                ),
            )
        ),
        wait=True,
    )
