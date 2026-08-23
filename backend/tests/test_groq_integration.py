import os
import pytest
from app.providers.groq import GroqProvider
from app.models.chat import ChatCompletionRequest, ChatMessage

# This test will be skipped if GROQ_API_KEY is not explicitly set in the environment
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="Requires real GROQ_API_KEY")
@pytest.mark.asyncio
async def test_groq_integration():
    # Make sure we use the environment variable
    from app.core.config import settings
    settings.GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    provider = GroqProvider()
    
    req = ChatCompletionRequest(
        model="llama-3.1-8b-instant",
        messages=[ChatMessage(role="user", content="Say 'Hello world' and nothing else.")],
        max_tokens=10,
        temperature=0.0
    )
    
    response = await provider.send_chat_completion(req)
    assert response.choices[0].message.content is not None
    assert "hello" in response.choices[0].message.content.lower()
