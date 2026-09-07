from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from rage.config import config
from rage.retriever import Retriever


def _metadata_id(collection_name: str) -> str:
    return str(uuid5(NAMESPACE_URL, collection_name))


async def set_collection_metadata(
    retriever: Retriever,
    collection_name: str,
    language: str,
    documents: list[str],
    files_md5: list[str],
) -> None:
    if not await retriever.qadrant_async_client.collection_exists(
        config.collection_metadata
    ):
        await retriever.qadrant_async_client.create_collection(
            collection_name=config.collection_metadata,
            vectors_config=models.VectorParams(
                size=1,
                distance=models.Distance.COSINE,
            ),
        )

    await retriever.qadrant_async_client.upsert(
        collection_name=config.collection_metadata,
        points=[
            models.PointStruct(
                id=_metadata_id(collection_name),
                vector=[1.0],
                payload={
                    "name": collection_name,
                    "language": language,
                    "documents": documents,
                    "files_md5": files_md5,
                },
            )
        ],
    )


async def get_collection_metadata(
    retriever: Retriever,
    collection_name: str,
) -> dict | None:
    if not await retriever.qadrant_async_client.collection_exists(
        config.collection_metadata
    ):
        return None

    records = await retriever.qadrant_async_client.retrieve(
        collection_name=config.collection_metadata,
        ids=[_metadata_id(collection_name)],
        with_payload=True,
    )
    return records[0].payload if records else None


async def remove_collection_metadata(
    retriever: Retriever,
    collection_name: str,
) -> None:
    if not await retriever.qadrant_async_client.collection_exists(
        config.collection_metadata
    ):
        return

    await retriever.qadrant_async_client.delete(
        collection_name=config.collection_metadata,
        points_selector=models.PointIdsList(
            points=[_metadata_id(collection_name)]
        ),
    )
