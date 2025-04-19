from datetime import time
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from main import app, request_timestamps

client = TestClient(app)

@pytest.mark.asyncio
@patch("main.forward_to_openrouter")
async def test_openrouter_chat(mock_forward):
    mock_forward.return_value = {
        "id": "test",
        "choices": [{"message": {"content": "Hello from OpenRouter"}}]
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json={
            "model": "deepseek/deepseek-r1:free",
            "messages": [{"role": "user", "content": "Hi"}]
        })

    assert response.status_code == 200
    assert "Hello from OpenRouter" in str(response.content)
    mock_forward.assert_called_once()


@pytest.mark.asyncio
@patch("main.forward_to_gemini")
async def test_gemini_chat(mock_forward):
    mock_forward.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}]
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Hi"}]
        })

    assert response.status_code == 200
    assert "Hello from Gemini" in str(response.content)
    mock_forward.assert_called_once()


@pytest.mark.asyncio
@patch("main.forward_to_openrouter")
async def test_rate_limit(mock_forward):
    mock_forward.return_value = {"choices": [{"message": {"content": "Hello"}}]}
    test_key = "openrouter_key_1"
    request_timestamps[test_key] = []

    # Simulate 61 requests to exceed rate limit (rate limit is 60/min by default)
    for _ in range(61):
        request_timestamps[test_key].append(time.time())

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json={
            "model": "deepseek/deepseek-r1:free",
            "messages": [{"role": "user", "content": "Hi"}]
        })

    assert response.status_code == 429
    assert "Rate limit exceeded" in response.text
