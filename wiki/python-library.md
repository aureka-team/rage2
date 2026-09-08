---
type: Reference
title: Python library
description: Main interfaces and components for indexing and search.
tags:
  - python
  - retrieval
---

# Python library

`rage.retriever.retriever.Retriever` is the main interface for indexing and
search. It:

- creates Qdrant collections with dense and sparse vectors;
- indexes `TextChunk` items into Qdrant;
- supports dense, sparse, hybrid, batch dense, and weighted dense search;
- supports Qdrant filters, score thresholds, scrolling, deletion, and payload
  indexes.

`rage.llm_agents.retrieval_assistant.RetrievalAssistant` answers a query using
only the relevant text chunks. The caller supplies both the query and serialized
chunks in `user_prompt`; its structured output contains `response=None` when
those chunks do not support an answer.

## Create and search a collection

Start Qdrant with `make qdrant-start` and set `OPENAI_API_KEY` before running
this example:

```python
import asyncio

from rage.meta.interfaces import Document
from rage.retriever import Retriever
from rage.splitters import MarkdownSplitter
from rage.utils.embeddings import get_openai_embeddings


async def main() -> None:
    collection_name = "example_documents"
    documents = [
        Document(
            text="# RAGE\n\nRAGE indexes documents and searches their text.",
            metadata={"document_id": "rage-overview"},
        )
    ]
    text_chunks = MarkdownSplitter().split_documents(documents=documents)
    retriever = Retriever(dense_embeddings=get_openai_embeddings())

    await retriever.create_collection(collection_name=collection_name)
    await retriever.insert_text_chunks(
        collection_name=collection_name,
        text_chunks=text_chunks,
    )

    results = await retriever.hybrid_search(
        collection_name=collection_name,
        query="What does RAGE do?",
        k=5,
    )

    for result in results:
        print(result.score, result.text)

    await retriever.qadrant_async_client.close()
    retriever.qadrant_client.close()


asyncio.run(main())
```

## Extending

Use the interfaces in `rage.meta.interfaces` for custom implementations:

- `TextLoader` is the base interface for document loaders.
- `TextSplitter` is the base interface for text splitters.

## Components

Current starting points include:

- `rage.retriever.retriever.Retriever`
- `rage.retriever.retriever.WeightedMetadataItem`
- `rage.llm_agents.retrieval_assistant.RetrievalAssistant`
- `rage.llm_agents.retrieval_assistant.RetrievalAssistantOutput`
- `rage.meta.interfaces.TextLoader`
- `rage.meta.interfaces.TextSplitter`
- `rage.loaders.aureka_transcription.AurekaTranscriptionLoader`
- `rage.loaders.pdf_markdown_loader.PDFMarkdownLoader`
- `rage.loaders.docx_loader.DocxLoader`
- `rage.loaders.markdown_loader.MarkdownLoader`
- `rage.splitters.document_splitter.DocumentSplitter`
- `rage.splitters.token_splitter.TokenSplitter`
- `rage.splitters.markdown_splitter.MarkdownSplitter`
