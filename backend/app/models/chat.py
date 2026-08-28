from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the author of this message. One of system, user, or assistant.")
    content: str = Field(..., description="The contents of the message.")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default", description="ID of the model to use.")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation so far.")
    temperature: Optional[float] = Field(default=1.0, description="What sampling temperature to use.")
    max_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate in the completion.")
    # Extensible generation parameters
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling.")

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ResponseMetadata(BaseModel):
    provider_name: str
    provider_model: str
    request_id: Optional[str] = None
    fallback_used: Optional[bool] = False
    cache_hit: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    provider: str
    choices: List[ChatChoice]
    usage: Optional[ChatCompletionUsage] = None
    metadata: ResponseMetadata
