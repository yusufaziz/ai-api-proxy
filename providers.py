from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
import logging
import json
import re
import time
from typing import AsyncGenerator

QUOTA_ERROR_PATTERNS = {
    "openrouter": [
        r"rate limit exceeded",
        r"quota exceeded",
        r"requests per (minute|day) exceeded"
    ],
    "gemini": [
        r"quota exceeded",
        r"resource exhausted",
        r"rate limit exceeded"
    ]
}

def update_rate_limits(provider: str, key: str, error_msg: str, rate_limiter):
    """Update rate limits when provider indicates quota exceeded"""
    patterns = QUOTA_ERROR_PATTERNS.get(provider, [])
    for pattern in patterns:
        if re.search(pattern, error_msg.lower()):
            # Force rate limit for this key
            rate_limit_key = f"req_day:{provider}:{key}"
            now = time.time()
            rate_limiter.rate_limit_windows[rate_limit_key] = [now] * rate_limiter.rate_limit_settings[provider]["max_request_day"]
            return True
    return False

async def format_stream_chunk(chunk) -> str:
    if hasattr(chunk.choices[0].delta, "tool_calls"):
        tool_calls = chunk.choices[0].delta.tool_calls
        chunk_data = {
            "choices": [{
                "index": chunk.choices[0].index,
                "delta": {"tool_calls": [
                    {
                        "index": tc.index,
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls if tool_calls
                ]}
            }]
        }
    else:
        chunk_data = {
            "choices": [{
                "index": chunk.choices[0].index,
                "delta": {"content": chunk.choices[0].delta.content or ""}
            }]
        }
    
    return f"data: {json.dumps(chunk_data)}\n\n"

async def stream_response(stream) -> AsyncGenerator[str, None]:
    try:
        async for chunk in stream:
            yield await format_stream_chunk(chunk)
    except Exception as e:
        logging.error(f"Streaming error: {e}")
    finally:
        yield "data: [DONE]\n\n"

async def call_openrouter_openai_compatible(payload: dict, key: str, rate_limiter=None):
    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1"
    )
    try:
        stream = payload.get("stream", False)
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            tools=payload.get("tools", None),
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                stream_response(response),
                media_type="text/event-stream"
            )
            
        return JSONResponse(status_code=200, content=response.model_dump())
    except Exception as e:
        error_msg = str(e)
        if rate_limiter and update_rate_limits("openrouter", key, error_msg, rate_limiter):
            return JSONResponse(
                status_code=429, 
                content={"error": f"Provider rate limit exceeded: {error_msg}"}
            )
        logging.error(f"OpenRouter error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to complete OpenRouter request: {error_msg}"})

async def call_gemini_openai_compatible(payload: dict, key: str, rate_limiter=None):
    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    try:
        stream = payload.get("stream", False)
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            tools=payload.get("tools", None),
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                stream_response(response),
                media_type="text/event-stream"
            )
            
        return JSONResponse(status_code=200, content=response.model_dump())
    except Exception as e:
        error_msg = str(e)
        if rate_limiter and update_rate_limits("gemini", key, error_msg, rate_limiter):
            return JSONResponse(
                status_code=429, 
                content={"error": f"Provider rate limit exceeded: {error_msg}"}
            )
        logging.error(f"Gemini error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to complete Gemini request: {error_msg}"})
