from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import json
import itertools
import logging
from typing import Dict, Optional
from collections import defaultdict

from rate_limiter import RateLimiter
from providers import call_openrouter_openai_compatible, call_gemini_openai_compatible

app = FastAPI()
rate_limiter = RateLimiter()

with open("config.json") as f:
    config = json.load(f)

provider_configs = config.get("provider_keys", [])
auto_models = config.get("auto-models", [])
access_key = config.get("access_key", "")
model_usage_gap_percentage = config.get("model_usage_gap_percentage", 5)

key_pools = defaultdict(itertools.cycle)
rate_limit_settings: Dict[str, Dict[str, int]] = {}

for provider in provider_configs:
    name = provider["provider"]
    keys = provider["key"]
    rate_limit_settings[name] = {
        "max_request_day": provider.get("max_request_day", 1500),
        "max_token_min": provider.get("max_token_min", 150000),
        "max_request_min": provider.get("max_request_min", 15)
    }
    key_pools[name] = itertools.cycle(keys)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_provider_from_model(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    return "openrouter"

def get_min_usage_key(provider: str) -> Optional[str]:
    usage = rate_limiter.request_counts[provider]
    if not usage:
        return None
    return min(usage.items(), key=lambda x: x[1])[0]

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing or invalid authorization header."})

    token = auth_header.split("Bearer ")[1]
    if token != access_key:
        return JSONResponse(status_code=403, content={"error": "Invalid access token."})

    body = await request.json()
    model = body.get("model", "")

    if model == "auto-model":
        model = await select_next_available_model()
        if not model:
            return JSONResponse(status_code=429, content={"error": "All models have reached their usage limits."})
        body["model"] = model

    # Ensure tools are properly passed if present
    if "tools" in body and not body["tools"]:
        del body["tools"]  # Remove empty tools array

    provider = get_provider_from_model(model)
    all_keys_tried = set()

    try:
        # First try the least used key
        min_usage_key = get_min_usage_key(provider)
        if min_usage_key and not rate_limiter.is_rate_limited(provider, min_usage_key, rate_limit_settings[provider]):
            key = min_usage_key
        else:
            # Try other available keys
            while len(all_keys_tried) < len(list(key_pools[provider])):
                key = next(key_pools[provider])
                if key in all_keys_tried:
                    continue
                    
                all_keys_tried.add(key)
                if not rate_limiter.is_rate_limited(provider, key, rate_limit_settings[provider]):
                    break
            else:
                return JSONResponse(
                    status_code=429, 
                    content={"error": f"All keys for provider '{provider}' have reached rate limits."}
                )

        logging.info(f"[REQUEST] Model: {model}, Provider: {provider}, Key: {key[:6]}***")

        if provider == "gemini":
            return await call_gemini_openai_compatible(body, key, rate_limiter)
        else:
            return await call_openrouter_openai_compatible(body, key, rate_limiter)

    except KeyError:
        return JSONResponse(
            status_code=500, 
            content={"error": f"No keys available for provider '{provider}'"}
        )

async def select_next_available_model() -> Optional[str]:
    for model in auto_models:
        provider = get_provider_from_model(model)
        for entry in provider_configs:
            if entry["provider"] == provider:
                # Try to get the least used key first
                min_usage_key = get_min_usage_key(provider)
                if min_usage_key and not rate_limiter.is_rate_limited(provider, min_usage_key, rate_limit_settings[provider]):
                    return model
                
                # If no min usage key available, try any non-rate-limited key
                for k in entry["key"]:
                    if not rate_limiter.is_rate_limited(provider, k, rate_limit_settings[provider]):
                        return model
    return None

@app.get("/v1/models")
async def list_models():
    models = []
    seen_ids = set()

    openrouter_entry = next((entry for entry in config["provider_keys"] if entry["provider"] == "openrouter"), None)
    openrouter_key = openrouter_entry["key"][0] if openrouter_entry and openrouter_entry["key"] else None

    if openrouter_key:
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {openrouter_key}"}
                )
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    for model in data:
                        model_id = model.get("id", "")
                        if model_id.endswith(":free"):
                            models.append({
                                "id": model_id,
                                "object": "model",
                                "owned_by": "openrouter"
                            })
                            seen_ids.add(model_id)
            except Exception:
                pass

    models.append({
        "id": "auto-model",
        "object": "model",
        "owned_by": "proxy"
    })

    return JSONResponse(status_code=200, content={"object": "list", "data": models})

@app.get("/v1/usage")
async def usage():
    return JSONResponse(status_code=200, content=rate_limiter.get_usage_data(rate_limit_settings))
