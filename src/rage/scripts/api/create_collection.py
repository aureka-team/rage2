import asyncio
import base64
import os

import requests

from rage.api.routers.collection.create.create_collection import (
    CollectionFile,
    CreateCollectionInput,
)
from rage.api.routers.collection.get.get_collection import GetCollectionInput
from rage.scripts.rendering import (
    console,
    render_header,
    render_step,
    render_step_detail,
)
from rage.utils.pdf import get_test_pdf

RAGE_API_URL = os.getenv("RAGE_API_URL", "http://rage-api:8000")
COLLECTION_NAME = "zaratustra"


def main() -> None:
    render_header()

    render_step("PDF download", "downloading Zaratustra")
    pdf_content = asyncio.run(get_test_pdf())
    render_step_detail("downloaded bytes", len(pdf_content))

    render_step("API collection", f"checking if {COLLECTION_NAME} exists")
    get_collection_input = GetCollectionInput(collection_name=COLLECTION_NAME)
    existing_collection_response = requests.post(
        f"{RAGE_API_URL}/rage/collection/get",
        json=get_collection_input.model_dump(mode="json"),
        timeout=30,
    )
    existing_collection_response.raise_for_status()
    collection_exists = (
        existing_collection_response.json()["collection_name"] is not None
    )
    render_step_detail("collection exists", collection_exists)
    if collection_exists:
        render_step(
            "API collection",
            f"deleting {COLLECTION_NAME} and its metadata",
        )

    render_step("API collection", f"creating {COLLECTION_NAME}")
    render_step_detail("API URL", RAGE_API_URL)
    create_collection_input = CreateCollectionInput(
        name=COLLECTION_NAME,
        collection_files=[
            CollectionFile(
                file_name="asi_hablo_zaratustra_nietzsche",
                file_type="application/pdf",
                base64_file=base64.b64encode(pdf_content).decode("ascii"),
            )
        ],
        language="es",
        overwrite=True,
    )

    api_response = requests.post(
        f"{RAGE_API_URL}/rage/collection/create",
        json=create_collection_input.model_dump(mode="json"),
        timeout=900,
    )
    api_response.raise_for_status()
    if collection_exists:
        render_step_detail("collection and metadata deleted", True)

    render_step_detail("status code", api_response.status_code)
    console.print_json(data=api_response.json())

    render_step("collection metadata", f"loading {COLLECTION_NAME}")
    metadata_response = requests.post(
        f"{RAGE_API_URL}/rage/collection/get",
        json=get_collection_input.model_dump(mode="json"),
        timeout=30,
    )
    metadata_response.raise_for_status()
    render_step_detail("status code", metadata_response.status_code)
    console.print_json(data=metadata_response.json())


if __name__ == "__main__":
    main()
