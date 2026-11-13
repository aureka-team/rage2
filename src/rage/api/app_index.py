"""Resolve a caller's app and its Qdrant collections."""

from dataclasses import dataclass
from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rage.config import config


@dataclass(frozen=True)
class AppIndex:
    content: str
    registry: str


_bearer = HTTPBearer(auto_error=False)


async def get_app_index(
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AppIndex:
    """Select the app from its configured bearer token."""
    app = "aureka"
    if config.rage_app_tokens:
        supplied = authorization.credentials.encode() if authorization else b""
        for app_id, token in config.rage_app_tokens.items():
            if compare_digest(supplied, token.get_secret_value().encode()):
                app = app_id
                break
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid RAGE app credential",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if app == "aureka":
        return AppIndex("rage_content_v1", f"{config.collection_metadata}_shared_v1")
    return AppIndex(f"rage_app_{app}_content_v1", f"rage_app_{app}_registry_v1")


AppIndexDependency = Annotated[AppIndex, Depends(get_app_index)]
