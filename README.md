# RAGE (RAG Engine)

RAGE is a Python toolkit for retrieval-augmented generation over local documents.
It provides loaders, splitters, embeddings, and a Qdrant-backed retriever for building indexing and search pipelines.

## Setup

Run the project inside the devcontainer. Jupyter is available there for the notebooks in [`notebooks/`](./notebooks).

Start Qdrant:

```bash
make qdrant-start
```

Optionally start Redis if you want to use loader caching:

```bash
make redis-start
```

## Environment variables

- `OPENAI_API_KEY`: required for OpenAI embeddings.
- `IONOS_TOKEN`: required for [`rage.embeddings.ionos_embeddings.IonosEmbeddings`](./rage/embeddings/ionos_embeddings.py).
- `QDRANT_HOST`: Qdrant host. Default: `localhost`.
- `QDRANT_PORT`: Qdrant HTTP port. Default: `6333`.
- `QDRANT_GRPC_PORT`: Qdrant gRPC port. Default: `6334`.
- `DENSE_EMBED_DOC_CACHE_PATH`: optional directory used to cache document embeddings during indexing.
- `DENSE_EMBED_QUERY_CACHE_PATH`: optional directory used to cache query embeddings during search.
- `FAST_EMBED_SPARSE_CACHE`: optional directory used by the sparse embedding model cache.

## Usage

The typical flow is:

1. Load documents.
2. Split them into chunks.
3. Create a Qdrant collection.
4. Insert chunks.
5. Query the retriever.

The example below uses the OpenAI helper from [`rage/utils/embeddings.py`](./rage/utils/embeddings.py). Any compatible embedding provider can be used.

```python
import asyncio

from rage.loaders.pdf_markdown_loader import PDFMarkdownLoader
from rage.retriever.retriever import Retriever
from rage.splitters.markdown_splitter import MarkdownSplitter
from rage.utils.embeddings import get_openai_embeddings


async def main() -> None:
    loader = PDFMarkdownLoader()
    documents = await loader.load("./resources/example.pdf")

    splitter = MarkdownSplitter(chunk_size=384, chunk_overlap=25)
    chunks = splitter.split_documents(documents)

    embeddings = get_openai_embeddings(
        model="text-embedding-3-large",
        dimensions=1024,
    )
    retriever = Retriever(dense_embeddings=embeddings)

    collection_name = "example_documents"
    await retriever.create_collection(collection_name)
    await retriever.insert_text_chunks(collection_name, chunks)

    results = await retriever.dense_search(
        collection_name=collection_name,
        query="What is this document about?",
        k=5,
    )

    for item in results:
        print(item.score, item.text[:200])


asyncio.run(main())
```

## Retriever

`rage.retriever.retriever.Retriever` is the main interface for indexing and search.

- Creates Qdrant collections with dense and sparse vectors.
- Indexes `TextChunk` items into Qdrant.
- Supports dense search, hybrid search, and batch dense search.
- Supports `dense_search_weighted` to boost dense results with weighted metadata matches.
- Supports Qdrant filters, score thresholds, scrolling, deletion, and payload indexes.

## Extending

Use the interfaces in `rage.meta.interfaces` to add custom implementations:

- `TextLoader`: base interface for custom document loaders.
- `TextSplitter`: base interface for custom text splitters.

## Components

Common starting points:

- `rage.retriever.retriever.Retriever`
- `rage.retriever.retriever.WeightedMetadataItem`
- `rage.meta.interfaces.TextLoader`
- `rage.meta.interfaces.TextSplitter`
- `rage.loaders.pdf_loader.PDFLoaeder`
- `rage.loaders.pdf_markdown_loader.PDFMarkdownLoader`
- `rage.loaders.docx_loader.DocxLoader`
- `rage.loaders.markdown_loader.MarkdownLoader`
- `rage.splitters.document_splitter.DocumentSplitter`
- `rage.splitters.token_splitter.TokenSplitter`
- `rage.splitters.markdown_splitter.MarkdownSplitter`
- `rage.splitters.title_splitter.TitleSplitter`
- `rage.splitters.word_splitter.WordSplitter`

## External installation

External projects can install `rage` as a regular dependency.

For example, [`lupai`](https://github.com/aureka-team/lupai) uses it as a dependency.

In `requirements.txt`:

```txt
rage>=<version>
```

In `uv.toml`:

```toml
[pip]
find-links = [
    "https://github.com/aureka-team/rage2/releases/expanded_assets/index",
]
```

## Releases

Pushing a Git tag matching `v*` creates a new release through [`.github/workflows/python-release.yml`](./.github/workflows/python-release.yml).

Example:

```bash
git tag v<version>
git push origin v<version>
```

The workflow builds the wheel, creates the GitHub release for the tag, and uploads the wheel to both the tag release and the permanent `index` release used by `uv`.

## Notebooks

Example notebooks are provided under [`notebooks/`](./notebooks):

- `01-retriever`
- `02-loaders`
- `03-embeddings`
- `04-converters`

## License

This project is licensed under the terms of the [`LICENSE`](./LICENSE) file.
