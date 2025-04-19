from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI, AsyncOpenAI
import logging
import json
import re
import time
from typing import AsyncGenerator
import traceback

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
    try:
        logging.debug(f"Processing chunk type: {type(chunk)}")
        if not chunk or not hasattr(chunk, 'choices') or not chunk.choices:
            logging.debug("Invalid chunk structure")
            raise ValueError("Invalid chunk structure")
            
        try:
            delta = chunk.choices[0].delta
            if not delta:
                logging.debug("Empty delta")
                raise ValueError("Empty delta")
        except Exception as e:
            logging.error(f"Error accessing delta: {e}")
            raise

        # For content type responses
        try:
            if hasattr(delta, "content"):
                content = delta.content
                if content:
                    chunk_data = {
                        "choices": [{
                            "index": chunk.choices[0].index,
                            "delta": {"content": content}
                        }]
                    }
                    return f"data: {json.dumps(chunk_data)}\n\n"
        except Exception as e:
            logging.error(f"Error processing content: {e}")
            raise
            
        # For tool call type responses
        try:
            if hasattr(delta, "tool_calls") and delta.tool_calls:
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
                            } for tc in delta.tool_calls  # Fixed tool_calls reference
                        ]}
                    }]
                }
                return f"data: {json.dumps(chunk_data)}\n\n"
        except Exception as e:
            logging.error(f"Error processing tool calls: {e}")
            raise
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Critical error formatting chunk: {str(e)}\nTraceback:\n{tb}")
        raise

async def stream_response(stream) -> AsyncGenerator[str, None]:
    chunk_count = 0
    last_yield_time = time.time()
    KEEPALIVE_INTERVAL = 5  # Reduced to 5 seconds
    TIMEOUT = 300  # 5 minutes total timeout
    
    try:
        async for chunk in stream:
            current_time = time.time()
            
            # Check total timeout
            if current_time - last_yield_time > TIMEOUT:
                logging.error("Stream timeout exceeded")
                yield f"data: {json.dumps({'error': 'Stream timeout'})}\n\n"
                break

            try:
                chunk_count += 1
                logging.debug(f"Processing chunk #{chunk_count}")
                
                formatted = await format_stream_chunk(chunk)
                if formatted:
                    yield formatted
                    last_yield_time = current_time
                else:
                    # Send keepalive on empty chunks or if too much time passed
                    if current_time - last_yield_time > KEEPALIVE_INTERVAL:
                        yield ":\n\n"  # SSE keepalive
                        last_yield_time = current_time
                        logging.debug("Sent keepalive ping")
                    
            except Exception as chunk_error:
                logging.error(f"Chunk error: {chunk_error}")
                # Don't break on chunk errors, try to continue
                if current_time - last_yield_time > KEEPALIVE_INTERVAL:
                    yield ":\n\n"
                    last_yield_time = current_time
                if isinstance(chunk_error, httpx.ReadError) and isinstance(chunk_error.args[0], httpx.ClosedResourceError):
                    logging.error("Stream closed by server")
                    break
                continue

    except Exception as e:
        logging.error(f"Stream error: {e}")
    finally:
        yield "data: [DONE]\n\n"

async def call_openrouter_openai_compatible(payload: dict, key: str, rate_limiter=None, stream: bool = False):
    logging.debug(f"OpenRouter request payload: {json.dumps(payload)}")
    client = AsyncOpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        timeout=300.0,  # Increased timeout to 5 minutes
        max_retries=3
    )
    try:
        response = await client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            tools=payload.get("tools", None),
            stream=stream,
            temperature=payload.get("temperature", 1.0),
            max_tokens=payload.get("max_tokens", None)
        )

        if stream:
            return StreamingResponse(
                stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no"
                }
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
    finally:
        pass

async def call_gemini_openai_compatible(payload: dict, key: str, rate_limiter=None, stream: bool = False):
    logging.debug(f"Gemini request payload: {json.dumps(payload)}")
    client = AsyncOpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=300.0,  # Increased timeout to 5 minutes
        max_retries=3
    )
    try:
        response = await client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            tools=payload.get("tools", None),
            stream=stream,
            temperature=payload.get("temperature", 0.7),
            max_tokens=payload.get("max_tokens", None)
        )

        if stream:
            return StreamingResponse(
                stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no"
                }
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
    finally:
        pass
