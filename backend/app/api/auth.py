from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.models.domain import ProjectAPIKey
from app.core.security import verify_api_key
from app.core.rate_limit import check_rate_limit
from fastapi import Request

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_current_api_key(
    request: Request,
    api_key_header: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> ProjectAPIKey:
    """
    Validates the API key from the Authorization header.
    Expects format: Bearer <token>
    """
    if not api_key_header:
        raise HTTPException(
            status_code=401,
            detail="Missing API Key"
        )
        
    # Handle optional "Bearer " prefix
    if api_key_header.startswith("Bearer "):
        raw_key = api_key_header[7:]
    else:
        raw_key = api_key_header
        
    # Usually we would extract a prefix to lookup quickly. 
    # For MVP without a structured prefix column strictly enforced, we'll scan active keys (inefficient for prod, but sufficient for now).
    # Since we added `api_key_prefix` to the model, we can extract the prefix to look it up!
    # Let's assume keys are "sk-aiflow-<token>", so prefix is first 15 chars.
    
    prefix = raw_key[:15]
    db_keys = db.query(ProjectAPIKey).filter(ProjectAPIKey.is_active == True).all()
    
    matched_key = None
    for db_key in db_keys:
        if verify_api_key(raw_key, db_key.hashed_key):
            matched_key = db_key
            break
            
    if not matched_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
        
    # Check rate limit
    check_rate_limit(request, str(matched_key.id))
    
    return matched_key
