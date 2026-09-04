import asyncio
import json
import uuid
from typing import Literal

from pydantic import BaseModel, NonNegativeFloat, StrictStr

from rage.meta.interfaces import Document, TextLoader


class DocumentMetadata(BaseModel):
    start: NonNegativeFloat | None = None
    end: NonNegativeFloat | None = None
    speaker: StrictStr | None = None
    source_id: uuid.UUID | None = None
    document_id: StrictStr | None = None
    document_type: Literal["transcription", "annotation"] | None = None
    block_id: StrictStr | None = None


class AurekaTranscriptionLoader(TextLoader):
    def __init__(self):
        super().__init__()

    def _get_documents(self, source_path: str) -> list[Document]:
        with open(source_path, "r") as f:
            transcription = json.loads(f.read())

        return [
            Document(
                text=item["text"],
                metadata=DocumentMetadata(**item).model_dump(exclude_none=True),
            )
            for item in transcription
        ]

    async def get_documents(
        self,
        source_path: str | None = None,
    ) -> list[Document]:
        if source_path is None:
            return []

        return await asyncio.to_thread(
            self._get_documents,
            source_path=source_path,
        )
