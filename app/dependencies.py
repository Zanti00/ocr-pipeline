import secrets

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

security = HTTPBearer(auto_error=False)


def resolve_service_from_token(token: str | None) -> str | None:
    """
    Constant-time lookup comparing a Bearer token against configured service API keys.
    Returns 'serms', 'prs', or None if invalid/missing.
    """
    if not token or not isinstance(token, str):
        return None

    if settings.serms_api_key and secrets.compare_digest(token, settings.serms_api_key):
        return "serms"
    if settings.prs_api_key and secrets.compare_digest(token, settings.prs_api_key):
        return "prs"

    return None


def verify_api_key(credentials: HTTPAuthorizationCredentials | None = Security(security)) -> str:
    """
    Validates the bearer token and returns the corresponding source_service (e.g. 'serms' or 'prs').
    Raises 403 Forbidden on missing or invalid API keys.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Missing or invalid API key.",
        )

    service = resolve_service_from_token(credentials.credentials)
    if service:
        return service

    raise HTTPException(
        status_code=403,
        detail="Forbidden: Invalid API key.",
    )

