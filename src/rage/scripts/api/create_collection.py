import base64
import os

import requests  # type: ignore
from rich.console import Console


ZARATUSTRA_PDF_URL = (
    "https://www.argentina.gob.ar/sites/default/files/"
    "asi_hablo_zaratustra_nietzsche.pdf"
)
RAGE_API_URL = os.getenv("RAGE_API_URL", "http://rage-api:8000")
COLLECTION_NAME = "zaratustra"

console = Console()


def main() -> None:
    console.log(f"Downloading {ZARATUSTRA_PDF_URL}")
    pdf_response = requests.get(ZARATUSTRA_PDF_URL, timeout=60)
    pdf_response.raise_for_status()
    console.log(f"Downloaded {len(pdf_response.content)} bytes")

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
                        pdf_response.content
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
