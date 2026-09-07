import os
from typing import Literal

import requests

from rage.api.routers.retriever.retrieve import RetrieverInput
from rage.scripts.rendering import (
    console,
    render_header,
    render_step,
    render_step_detail,
)

RAGE_API_URL = os.getenv("RAGE_API_URL", "http://rage-api:8000")
COLLECTION_NAME = "zaratustra"
SEMANTIC_QUERY = "Que quiere el gran Dragón?"
KEYWORD_QUERY = "dragon escama gran"
RETRIEVER_LIMIT = 5
MIN_SIMILARITY = 0.3
SEARCH_QUERIES: dict[Literal["dense", "hybrid", "sparse"], str] = {
    "dense": SEMANTIC_QUERY,
    "hybrid": SEMANTIC_QUERY,
    "sparse": KEYWORD_QUERY,
}


def main() -> None:
    render_header()

    render_step_detail("API URL", RAGE_API_URL)
    for search_mode, query in SEARCH_QUERIES.items():
        console.rule(style="dim magenta")
        render_step(
            f"{search_mode} retrieval",
            f"searching {COLLECTION_NAME}",
        )
        render_step_detail("query", query)
        retriever_input = RetrieverInput(
            collection_names=[COLLECTION_NAME],
            query_text=query,
            search_mode=search_mode,
            retriever_limit=RETRIEVER_LIMIT,
            min_similarity=MIN_SIMILARITY,
            enable_reranker=True,
        )

        api_response = requests.post(
            f"{RAGE_API_URL}/rage/retriever/retrieve",
            json=retriever_input.model_dump(mode="json"),
            timeout=60,
        )
        api_response.raise_for_status()

        response = api_response.json()
        render_step_detail("status code", api_response.status_code)
        render_step_detail(
            "items retrieved",
            len(response["retriever_items"]),
        )
        console.print_json(data=response, ensure_ascii=False)


if __name__ == "__main__":
    main()
