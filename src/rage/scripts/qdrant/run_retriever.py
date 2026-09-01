import asyncio
import tempfile

from langchain_openai import OpenAIEmbeddings
from rich.text import Text

from rage.config import config
from rage.loaders import PDFMarkdownLoader
from rage.meta.interfaces import Document, TextChunk
from rage.retriever import Retriever, RetrieverItem
from rage.scripts.rendering import (
    console,
    render_header,
    render_step,
    render_step_detail,
)
from rage.splitters import MarkdownSplitter
from rage.utils.pdf import get_test_pdf

SEMANTIC_QUERY = "Que quiere el gran Dragón?"
KEYWORD_QUERY = "dragon escama gran"
COLLECTION_NAME = "rage_test"
MAX_ITEMS = 5
SCORE_THRESHOLD = 0.3


def render_collection_details(
    collection_name: str,
    documents: list[Document],
    text_chunks: list[TextChunk],
) -> None:
    message = Text()
    message.append("\n┌─[ ", style="dim magenta")
    message.append("COLLECTION CREATED ]\n", style="bold white")
    message.append("├── ", style="dim magenta")
    message.append("NAME // ", style="dim white")
    message.append(f"{collection_name}\n", style="bold white")
    message.append("├── ", style="dim magenta")
    message.append("DOCUMENTS // ", style="dim white")
    message.append(f"{len(documents)}\n", style="bold white")
    message.append("└── ", style="dim magenta")
    message.append("CHUNKS // ", style="dim white")
    message.append(f"{len(text_chunks)}\n", style="bold white")
    console.print(message)


def render_retrieval_details(
    search_type: str,
    query: str,
    items: list[RetrieverItem],
) -> None:
    message = Text()
    message.append("\n┌─[ ", style="dim magenta")
    message.append(f"{search_type.upper()} RETRIEVAL ]\n", style="bold white")
    message.append("├── ", style="dim magenta")
    message.append("QUERY // ", style="dim white")
    message.append(f"{query}\n", style="dim white")
    message.append("└──> ", style="dim magenta")
    message.append(f"{len(items)} ITEMS RETRIEVED\n", style="dim white")
    console.print(message)
    console.print("[bold white]ITEMS[/bold white]")
    console.print_json(
        data=[item.model_dump(mode="json") for item in items],
        ensure_ascii=False,
        indent=2,
    )


async def get_zaratustra_documents() -> list[Document]:
    render_step("PDF download", "downloading Zaratustra")
    pdf_content = await get_test_pdf()
    render_step_detail("downloaded bytes", len(pdf_content))

    with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
        pdf_file.write(pdf_content)
        pdf_file.flush()

        render_step("PDF markdown loader", "loading PDF documents")
        return await PDFMarkdownLoader().load(
            source_path=pdf_file.name,
            cached_load=True,
        )


async def main() -> None:
    render_header()

    render_step("document loader", "loading or restoring cached documents")
    documents = await get_zaratustra_documents()
    render_step_detail("documents loaded", len(documents))

    render_step("markdown splitter", "splitting documents into chunks")
    text_chunks = MarkdownSplitter().split_documents(documents=documents)
    render_step_detail("chunks created", len(text_chunks))

    render_step("retriever", "initializing dense and sparse embeddings")
    retriever = Retriever(
        dense_embeddings=OpenAIEmbeddings(
            model=config.emb_model,
            dimensions=config.emb_dimensions,
        )
    )
    render_step_detail("dense model", config.emb_model)
    render_step_detail("dense dimensions", config.emb_dimensions)

    render_step("Qdrant collection", f"checking if {COLLECTION_NAME} exists")
    collection_exists = await retriever.qadrant_async_client.collection_exists(
        collection_name=COLLECTION_NAME
    )
    render_step_detail("collection exists", collection_exists)

    if collection_exists:
        render_step("Qdrant collection", f"deleting {COLLECTION_NAME}")
        await retriever.qadrant_async_client.delete_collection(
            collection_name=COLLECTION_NAME
        )
        render_step_detail("collection deleted", True)

    render_step("Qdrant collection", f"creating {COLLECTION_NAME}")
    await retriever.create_collection(collection_name=COLLECTION_NAME)

    render_step("Qdrant collection", "inserting text chunks")
    await retriever.insert_text_chunks(
        collection_name=COLLECTION_NAME,
        text_chunks=text_chunks,
    )
    render_collection_details(
        collection_name=COLLECTION_NAME,
        documents=documents,
        text_chunks=text_chunks,
    )

    console.rule(style="dim magenta")
    render_step("dense retrieval", "running dense search")
    dense_results = await retriever.dense_search(
        collection_name=COLLECTION_NAME,
        query=SEMANTIC_QUERY,
        k=MAX_ITEMS,
        score_threshold=SCORE_THRESHOLD,
    )
    render_retrieval_details("dense", SEMANTIC_QUERY, dense_results)

    console.rule(style="dim magenta")
    render_step("hybrid retrieval", "running hybrid search")
    hybrid_results = await retriever.hybrid_search(
        collection_name=COLLECTION_NAME,
        query=SEMANTIC_QUERY,
        k=MAX_ITEMS,
    )
    render_retrieval_details("hybrid", SEMANTIC_QUERY, hybrid_results)

    console.rule(style="dim magenta")
    render_step("sparse retrieval", "running sparse search")
    sparse_results = await retriever.sparse_search(
        collection_name=COLLECTION_NAME,
        query=KEYWORD_QUERY,
        k=MAX_ITEMS,
    )
    render_retrieval_details("sparse", KEYWORD_QUERY, sparse_results)

    render_step("Qdrant clients", "closing client connections")
    await retriever.qadrant_async_client.close()
    retriever.qadrant_client.close()
    render_step_detail("clients closed", True)


if __name__ == "__main__":
    asyncio.run(main())
