from asyncio import Lock
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from rage.retriever import Retriever

_setup_lock = Lock()


def _metadata_id(collection_name: str) -> str:
    return str(uuid5(NAMESPACE_URL, collection_name))


async def set_collection_metadata(
    retriever: Retriever,
    registry: str,
    collection_name: str,
    language: str,
    documents: list[str],
    files_md5: list[str],
    generation: str,
) -> None:
    async with _setup_lock:
        if not await retriever.qadrant_async_client.collection_exists(registry):
            await retriever.qadrant_async_client.create_collection(
                collection_name=registry,
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE
                ),
            )

    await retriever.qadrant_async_client.upsert(
        collection_name=registry,
        points=[
            models.PointStruct(
                id=_metadata_id(collection_name),
                vector=[1.0],
                payload={
                    "name": collection_name,
                    "language": language,
                    "documents": documents,
                    "files_md5": files_md5,
                    "generation": generation,
                },
            )
        ],
        wait=True,
    )


async def get_collection_metadata(
    retriever: Retriever,
    registry: str,
    collection_name: str,
) -> dict | None:
    return (await get_resource_metadata(retriever, registry, [collection_name])).get(
        collection_name
    )


async def get_resource_metadata(
    retriever: Retriever, registry: str, names: list[str]
) -> dict[str, dict]:
    client = retriever.qadrant_async_client
    if not names or not await client.collection_exists(registry):
        return {}

    records = await client.retrieve(
        collection_name=registry,
        ids=[_metadata_id(name) for name in dict.fromkeys(names)],
        with_payload=True,
    )
    return {
        record.payload["name"]: record.payload for record in records if record.payload
    }


async def list_resource_names(retriever: Retriever, registry: str) -> list[str]:
    client = retriever.qadrant_async_client
    if not await client.collection_exists(registry):
        return []

    names = []
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=registry, with_payload=["name"], offset=offset
        )
        names.extend(record.payload["name"] for record in records if record.payload)
        if offset is None:
            return sorted(names)


async def remove_collection_metadata(
    retriever: Retriever,
    registry: str,
    collection_name: str,
) -> None:
    if not await retriever.qadrant_async_client.collection_exists(registry):
        return

    await retriever.qadrant_async_client.delete(
        collection_name=registry,
        points_selector=models.PointIdsList(points=[_metadata_id(collection_name)]),
        wait=True,
    )
