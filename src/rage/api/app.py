from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from rage.api.routers import (
    collection_create_router,
    collection_get_objects_router,
    collection_get_router,
    collection_list_router,
    collection_remove_router,
    retrieve_router,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.get("/", include_in_schema=False)(lambda: RedirectResponse(url="/docs/"))


@app.get(
    "/healthcheck",
    tags=["status"],
    summary="Check API health",
    description="Returns the current availability status of the RAGE API.",
)
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(collection_create_router)
app.include_router(collection_remove_router)
app.include_router(collection_list_router)
app.include_router(collection_get_router)
app.include_router(collection_get_objects_router)
app.include_router(retrieve_router)


__all__ = ["app"]
