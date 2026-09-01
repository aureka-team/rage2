import base64
import binascii
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    StrictBool,
    StrictStr,
    field_validator,
)
from pydantic_extra_types.language_code import LanguageAlpha2

from rage.api.utils import get_retriever
from rage.loaders import PDFMarkdownLoader
from rage.meta.interfaces import Document, TextChunk
from rage.retriever import Retriever
from rage.splitters import MarkdownSplitter

VALID_FILE_TYPES = {"application/json", "application/pdf", "text/plain"}


class CollectionFile(BaseModel):
    file_name: StrictStr = Field(
        description="Name used to identify the source document in the collection."
    )
    file_type: StrictStr = Field(
        description="MIME type of the encoded file content."
    )
    base64_file: StrictStr = Field(
        description="Complete file content encoded as a Base64 string."
    )

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, value: str) -> str:
        if value in VALID_FILE_TYPES:
            return value

        raise ValueError(f"valid file types are: {sorted(VALID_FILE_TYPES)}")


class CreateCollectionInput(BaseModel):
    name: StrictStr = Field(
        description=(
            "Collection name. It is normalized to lowercase with whitespace "
            "replaced by underscores."
        )
    )
    collection_files: list[CollectionFile] = Field(
        description="Files to load, split, embed, and store in the collection."
    )
    language: LanguageAlpha2 = Field(
        description="ISO 639-1 language code for the collection content.",
    )
    overwrite: StrictBool = Field(
        default=False,
        description="Delete and recreate the collection when it already exists.",
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return "_".join(value.lower().split())


class GraphStats(BaseModel):
    num_nodes: PositiveInt = Field(
        description="Number of nodes in the collection hierarchy."
    )
    num_edges: PositiveInt = Field(
        description="Number of edges in the collection hierarchy."
    )


class CreateCollectionOutput(BaseModel):
    collection_name: StrictStr = Field(
        description="Normalized name of the collection."
    )
    collection_language: StrictStr = Field(
        description="ISO 639-1 language code stored for the collection."
    )
    collection_documents: list[StrictStr] = Field(
        description="Names of the source documents assigned to the collection."
    )
    created: StrictBool = Field(
        default=False,
        description="Whether a new collection was created by this request.",
    )
    num_documents: PositiveInt | None = Field(
        default=None,
        description="Number of documents loaded into the new collection.",
    )
    num_leaf_text_chunks: PositiveInt | None = Field(
        default=None,
        description="Number of text chunks inserted into the collection.",
    )
    graph_stats: GraphStats | None = Field(
        default=None,
        description="Hierarchy statistics when hierarchical indexing is available.",
    )


def _decode_file(collection_file: CollectionFile) -> bytes:
    try:
        return base64.b64decode(collection_file.base64_file, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(
            status_code=422,
            detail=f"invalid base64 content for {collection_file.file_name}",
        ) from error


async def _load_file(
    collection_file: CollectionFile,
    temporary_directory: str,
) -> list[Document]:
    file_bytes = _decode_file(collection_file)
    if collection_file.file_type == "text/plain":
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=422,
                detail=f"{collection_file.file_name} is not valid UTF-8",
            ) from error

        return [
            Document(
                text=text, metadata={"document_name": collection_file.file_name}
            )
        ]

    if collection_file.file_type == "application/json":
        raise HTTPException(
            status_code=501,
            detail="application/json audio transcriptions are not supported",
        )

    file_path = Path(temporary_directory) / f"{collection_file.file_name}.pdf"
    file_path.write_bytes(file_bytes)
    documents = await PDFMarkdownLoader().load(
        source_path=str(file_path),
        cached_load=True,
    )
    return [
        Document(
            text=document.text,
            metadata=document.metadata
            | {"document_name": collection_file.file_name},
        )
        for document in documents
    ]


collection_create_router = APIRouter()


@collection_create_router.post(
    "/rage/collection/create",
    tags=["collection"],
    summary="Create a collection",
    description=(
        "Decodes the supplied files, extracts and splits their text, then creates "
        "a Qdrant collection containing the resulting embeddings."
    ),
)
async def create_collection(
    request: CreateCollectionInput,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> CreateCollectionOutput:
    exists = await retriever.qadrant_async_client.collection_exists(
        request.name
    )
    if exists and not request.overwrite:
        return CreateCollectionOutput(
            collection_name=request.name,
            collection_language=request.language,
            collection_documents=[
                item.file_name for item in request.collection_files
            ],
        )

    if exists:
        await retriever.qadrant_async_client.delete_collection(request.name)

    with TemporaryDirectory() as temporary_directory:
        document_groups = [
            await _load_file(item, temporary_directory)
            for item in request.collection_files
        ]

    documents = [document for group in document_groups for document in group]
    chunks = MarkdownSplitter().split_documents(documents)
    collection_documents = [item.file_name for item in request.collection_files]
    chunks = [
        TextChunk(
            text=chunk.text,
            metadata=chunk.metadata
            | {
                "collection_language": request.language,
                "collection_documents": collection_documents,
            },
            num_tokens=chunk.num_tokens,
        )
        for chunk in chunks
    ]

    await retriever.create_collection(request.name)
    await retriever.insert_text_chunks(request.name, chunks)
    return CreateCollectionOutput(
        collection_name=request.name,
        collection_language=request.language,
        collection_documents=collection_documents,
        created=True,
        num_documents=len(documents),
        num_leaf_text_chunks=len(chunks),
    )
