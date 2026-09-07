import asyncio

from langchain_openai import OpenAIEmbeddings

from rage.config import config
from rage.retriever import Retriever
from rage.scripts.rendering import (
    console,
    render_header,
    render_step,
    render_step_detail,
)

COLLECTION_NAME = "zaratustra"
CENTER_CHUNK_INDEX = 2
BEFORE = 5
AFTER = 5


async def main() -> None:
    render_header()

    render_step("retriever", "initializing embeddings")
    retriever = Retriever(
        dense_embeddings=OpenAIEmbeddings(
            model=config.emb_model,
            dimensions=config.emb_dimensions,
        )
    )
    render_step_detail("dense model", config.emb_model)
    render_step_detail("dense dimensions", config.emb_dimensions)

    render_step(
        "text chunk",
        f"loading chunk index {CENTER_CHUNK_INDEX} from {COLLECTION_NAME}",
    )
    center = await retriever.get_text_chunk(
        collection_name=COLLECTION_NAME,
        metadata_key="metadata.chunk_index",
        metadata_value=CENTER_CHUNK_INDEX,
    )

    if center is None:
        render_step_detail("chunk found", False)
        await retriever.qadrant_async_client.close()
        retriever.qadrant_client.close()
        return

    chunk_id = center.metadata["chunk_id"]
    assert isinstance(chunk_id, str)
    render_step_detail("chunk found", True)
    render_step_detail("chunk ID", chunk_id)

    render_step(
        "neighboring chunks",
        f"loading {BEFORE} before and {AFTER} after",
    )
    items = await retriever.get_neighboring_text_chunks(
        collection_name=COLLECTION_NAME,
        chunk_id=chunk_id,
        before=BEFORE,
        after=AFTER,
    )
    render_step_detail("items retrieved", len(items))
    console.print_json(
        data=[item.model_dump(mode="json") for item in items],
        ensure_ascii=False,
        indent=2,
    )

    render_step("Qdrant clients", "closing client connections")
    await retriever.qadrant_async_client.close()
    retriever.qadrant_client.close()
    render_step_detail("clients closed", True)


if __name__ == "__main__":
    asyncio.run(main())
