import os
import pytest
from app.providers.gemini import GeminiProvider
from app.models.chat import ChatCompletionRequest, ChatMessage

# This test will be skipped if GEMINI_API_KEY is not explicitly set in the environment
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="Requires real GEMINI_API_KEY")
@pytest.mark.asyncio
async def test_gemini_integration():
    # Make sure we use the environment variable
    from app.core.config import settings
    settings.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    provider = GeminiProvider()
    
    req = ChatCompletionRequest(
        model="gemini-1.5-flash",
        messages=[ChatMessage(role="user", content="Say 'Hello world' and nothing else.")],
        max_tokens=10,
        temperature=0.0
    )
    
    response = await provider.send_chat_completion(req)
    assert response.choices[0].message.content is not None
    assert "hello" in response.choices[0].message.content.lower()
