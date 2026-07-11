from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validates the bearer token and returns the corresponding source_service (e.g. 'serms' or 'prs').
    """
    token = credentials.credentials
    
    if token == settings.serms_api_key:
        return "serms"
    elif token == settings.prs_api_key:
        return "prs"
    
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key."
    )
