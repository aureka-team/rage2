import asyncio
import base64
import os

import requests
from rich.console import Console

from rage.utils.pdf import get_test_pdf

RAGE_API_URL = os.getenv("RAGE_API_URL", "http://rage-api:8000")
COLLECTION_NAME = "zaratustra"

console = Console()


def main() -> None:
    console.log("Loading test PDF")
    pdf_content = asyncio.run(get_test_pdf())
    console.log(f"Loaded {len(pdf_content)} bytes")

    console.log(f"Creating {COLLECTION_NAME} through {RAGE_API_URL}")
    api_response = requests.post(
        f"{RAGE_API_URL}/rage/collection/create",
        json={
            "name": COLLECTION_NAME,
            "collection_files": [
                {
                    "file_name": "asi_hablo_zaratustra_nietzsche",
                    "file_type": "application/pdf",
                    "base64_file": base64.b64encode(
                        pdf_content
                    ).decode("ascii"),
                }
            ],
            "language": "es",
            "overwrite": True,
        },
        timeout=900,
    )
    api_response.raise_for_status()
    console.print_json(data=api_response.json())


if __name__ == "__main__":
    main()
