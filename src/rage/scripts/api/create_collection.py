import asyncio
import base64
import os
from itertools import cycle

import requests
from rich.align import Align
from rich.console import Console
from rich.text import Text

from rage.utils.pdf import get_test_pdf

RAGE_API_URL = os.getenv("RAGE_API_URL", "http://rage-api:8000")
COLLECTION_NAME = "zaratustra"
RAGE_BANNER = (
    "██████╗  █████╗  ██████╗ ███████╗",
    "██╔══██╗██╔══██╗██╔════╝ ██╔════╝",
    "██████╔╝███████║██║  ███╗█████╗  ",
    "██╔══██╗██╔══██║██║   ██║██╔══╝  ",
    "██║  ██║██║  ██║╚██████╔╝███████╗",
    "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
)
RAGE_STYLES = (
    "bold bright_magenta",
    "bold magenta",
    "bold bright_cyan",
)

console = Console()


def render_header() -> None:
    banner = Text()
    banner.append("\n\n")

    for line, style in zip(RAGE_BANNER, cycle(RAGE_STYLES), strict=False):
        banner.append(f"{line}\n", style=style)

    banner.append(
        "R A G   E N G I N E".center(len(RAGE_BANNER[0])),
        style="dim bright_magenta",
    )
    console.print(Align.center(banner))


def render_step(label: str, action: str) -> None:
    message = Text()
    message.append("\n┌─[ ", style="dim magenta")
    message.append(f"{label.upper()} ]\n", style="bold white")
    message.append("└──> ", style="dim magenta")
    message.append(f"{action.upper()}...\n", style="dim white")
    console.print(message)


def render_step_detail(label: str, value: object) -> None:
    detail = Text()
    detail.append(" :: ", style="dim magenta")
    detail.append(label.upper(), style="bold white")
    detail.append(" // ", style="dim magenta")
    detail.append(str(value), style="dim white")
    console.print(detail)


def main() -> None:
    render_header()

    render_step("PDF download", "downloading Zaratustra")
    pdf_content = asyncio.run(get_test_pdf())
    render_step_detail("downloaded bytes", len(pdf_content))

    render_step("API collection", f"checking if {COLLECTION_NAME} exists")
    existing_collection_response = requests.post(
        f"{RAGE_API_URL}/rage/collection/get",
        json={"collection_name": COLLECTION_NAME},
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
    if collection_exists:
        render_step_detail("collection and metadata deleted", True)

    render_step_detail("status code", api_response.status_code)
    console.print_json(data=api_response.json())

    render_step("collection metadata", f"loading {COLLECTION_NAME}")
    metadata_response = requests.post(
        f"{RAGE_API_URL}/rage/collection/get",
        json={"collection_name": COLLECTION_NAME},
        timeout=30,
    )
    metadata_response.raise_for_status()
    render_step_detail("status code", metadata_response.status_code)
    console.print_json(data=metadata_response.json())


if __name__ == "__main__":
    main()
