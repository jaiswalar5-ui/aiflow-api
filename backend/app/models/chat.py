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
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatCompletionUsage
