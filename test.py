import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json
import os
from fastapi.responses import JSONResponse, StreamingResponse
from main import app

# Ensure config.json exists for tests
TEST_CONFIG = {
    "access_key": "test_access_key",
    "model_usage_gap_percentage": 5,
    "auto-models": ["gpt-3.5-turbo:free", "gemini-pro"],
    "provider_keys": [
        {
            "provider": "openrouter",
            "key": ["test_or_key1", "test_or_key2"],
            "max_request_day": 1500,
            "max_token_min": 150000,
            "max_request_min": 15
        },
        {
            "provider": "gemini",
            "key": ["test_g_key1", "test_g_key2"],
            "max_request_day": 1500,
            "max_token_min": 150000,
            "max_request_min": 15
        }
    ]
}

@pytest.fixture(autouse=True)
async def mock_dependencies():
    with patch('main.config', TEST_CONFIG), \
         patch('main.rate_limiter') as mock_limiter, \
         patch('providers.call_openrouter_openai_compatible') as mock_or, \
         patch('providers.call_gemini_openai_compatible') as mock_gemini:
            
        mock_limiter.is_rate_limited.return_value = False
        mock_limiter.get_usage_data.return_value = {
            "overview": {
                "openrouter": {
                    "total_requests": 100,
                    "total_capacity": 3000,
                    "usage_percentage": 3.33
                }
            },
            "details": {}
        }
        
        mock_response = JSONResponse(
            status_code=200,
            content={
                "id": "test",
                "choices": [{"message": {"content": "Hello"}}]
            }
        )
        mock_or.return_value = mock_response
        mock_gemini.return_value = mock_response
        
        yield {
            "limiter": mock_limiter,
            "openrouter": mock_or,
            "gemini": mock_gemini
        }

@pytest.mark.asyncio
async def test_chat_unauthorized():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json={})
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_openrouter_chat(mock_dependencies):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"},
            json={
                "model": "gpt-3.5-turbo:free",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
    
    assert response.status_code == 200
    response_data = response.json()
    assert "choices" in response_data
    assert response_data["choices"][0]["message"]["content"] == "Hello"

@pytest.mark.asyncio
async def test_auto_model_selection(mock_dependencies):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"},
            json={
                "model": "auto-model",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
    assert response.status_code == 200
    assert mock_dependencies["openrouter"].called or mock_dependencies["gemini"].called

@pytest.mark.asyncio
@patch('providers.call_openrouter_openai_compatible')
async def test_tools_request(mock_call, mock_dependencies):
    mock_call.return_value = JSONResponse(
        status_code=200,
        content={
            "id": "test",
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"name": "get_weather", "arguments": '{"location":"London"}'}
                    }]
                }
            }]
        }
    )

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"},
            json={
                "model": "gpt-3.5-turbo:free",
                "messages": [{"role": "user", "content": "Weather in London?"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}}
                        }
                    }
                }]
            }
        )
    
    assert response.status_code == 200
    assert "tool_calls" in response.json()["choices"][0]["message"]

@pytest.mark.asyncio
async def test_models_endpoint(mock_dependencies):
    with patch('httpx.AsyncClient.get') as mock_http_get:
        mock_http_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "gpt-3.5-turbo:free", "object": "model"},
                    {"id": "anthropic/claude-2:free", "object": "model"}
                ]
            }
        )

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"}
            )
        
        assert response.status_code == 200
        models = response.json()["data"]
        assert any(model["id"] == "auto-model" for model in models)
        assert any(model["id"] == "gpt-3.5-turbo:free" for model in models)

@pytest.mark.asyncio
async def test_rate_limit_exceeded(mock_dependencies):
    mock_dependencies["limiter"].is_rate_limited.return_value = True

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"},
            json={
                "model": "gpt-3.5-turbo:free",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
    
    assert response.status_code == 429
    assert "rate limits" in response.json()["error"].lower()

@pytest.mark.asyncio
@patch('providers.call_openrouter_openai_compatible')
async def test_streaming(mock_call, mock_dependencies):
    async def mock_stream():
        yield b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
        yield b"data: [DONE]\n\n"

    mock_response = StreamingResponse(mock_stream(), media_type="text/event-stream")
    mock_call.return_value = mock_response

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"},
            json={
                "model": "gpt-3.5-turbo:free",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True
            }
        )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"

@pytest.mark.asyncio
async def test_usage_endpoint(mock_dependencies):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(
            "/v1/usage",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "overview" in data

@pytest.mark.asyncio
async def test_provider_quota_exceeded(mock_dependencies):
    mock_dependencies["openrouter"].return_value = JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded: quota exceeded for this minute"}
    )

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"},
            json={
                "model": "gpt-3.5-turbo:free",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
    
    assert response.status_code == 429
    assert "provider rate limit exceeded" in response.json()["error"].lower()
    # Check if rate limiter was updated
    assert mock_dependencies["limiter"].rate_limit_windows.__setitem__.called

@pytest.mark.asyncio
async def test_provider_error_handling(mock_dependencies):
    mock_dependencies["openrouter"].return_value = JSONResponse(
        status_code=500,
        content={"error": "Some other error"}
    )

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_CONFIG['access_key']}"},
            json={
                "model": "gpt-3.5-turbo:free",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
    
    assert response.status_code == 500
    assert "failed to complete" in response.json()["error"].lower()
