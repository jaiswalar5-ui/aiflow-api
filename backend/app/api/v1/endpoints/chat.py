import time
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.auth import get_current_api_key
from app.models.domain import ProjectAPIKey
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, ChatCompletionUsage

router = APIRouter()

@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: ProjectAPIKey = Depends(get_current_api_key)
):
    """
    Endpoint for chat completions.
    Routes the request to the appropriate AI provider.
    """
    from app.services.router import router_engine
    from fastapi import HTTPException
    
    # In the future, target_provider could be determined by headers or routing logic.
    # For MVP, it routes to the default provider if target_provider=None.
    return await router_engine.route_chat_completion(request)
