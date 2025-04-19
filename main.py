from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import json
import itertools
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from openai import OpenAI

app = FastAPI()

with open("config.json") as f:
    config = json.load(f)

provider_configs = config.get("provider_keys", [])
auto_models = config.get("auto-models", [])
access_key = config.get("access_key", "")
model_usage_gap_percentage = config.get("model_usage_gap_percentage", 5)

key_pools = defaultdict(itertools.cycle)
request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
rate_limit_windows: Dict[str, List[float]] = defaultdict(list)
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

def is_rate_limited(provider: str, key: str) -> bool:
    now = time.time()
    window_min = 60
    settings = rate_limit_settings[provider]

    # Per-minute request limit
    req_min_window = [t for t in rate_limit_windows[f"req_min:{provider}:{key}"] if t > now - window_min]
    if len(req_min_window) >= settings["max_request_min"]:
        return True
    rate_limit_windows[f"req_min:{provider}:{key}"] = req_min_window + [now]

    # Placeholder for token logic (can be enhanced if tokens are known)
    # Per-minute token limit not applied here directly due to lack of token count

    # Per-day request limit
    req_day_window = [t for t in rate_limit_windows[f"req_day:{provider}:{key}"] if t > now - 86400]
    if len(req_day_window) >= settings["max_request_day"]:
        return True
    rate_limit_windows[f"req_day:{provider}:{key}"] = req_day_window + [now]

    request_counts[provider][key] += 1
    return False

def get_provider_from_model(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    return "openrouter"

def usage_gap_exceeded(provider: str) -> bool:
    usage = request_counts[provider]
    values = list(usage.values())
    if len(values) < 2:
        return False
    max_val = max(values)
    min_val = min(values)
    if max_val == 0:
        return False
    gap_percent = ((max_val - min_val) / max_val) * 100
    return gap_percent > model_usage_gap_percentage

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

    provider = get_provider_from_model(model)

    try:
        for _ in range(len(provider_configs)):
            key = next(key_pools[provider])
            if not is_rate_limited(provider, key):
                if not usage_gap_exceeded(provider):
                    break
        else:
            return JSONResponse(status_code=429, content={"error": f"No usable key for provider '{provider}' within usage gap limit."})
    except KeyError:
        return JSONResponse(status_code=500, content={"error": f"No keys available for provider '{provider}'"})

    logging.info(f"[REQUEST] Model: {model}, Provider: {provider}, Key: {key[:6]}***")

    if provider == "gemini":
        return await call_gemini_openai_compatible(body, key)
    else:
        return await call_openrouter_openai_compatible(body, key)

async def select_next_available_model() -> Optional[str]:
    for model in auto_models:
        provider = get_provider_from_model(model)
        for entry in provider_configs:
            if entry["provider"] == provider:
                for k in entry["key"]:
                    if not is_rate_limited(provider, k):
                        if not usage_gap_exceeded(provider):
                            return model
    return None

async def call_openrouter_openai_compatible(payload: dict, key: str):
    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1"
    )
    try:
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"]
        )
        return JSONResponse(status_code=200, content=response.model_dump())
    except Exception as e:
        logging.error(f"OpenRouter error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to complete OpenRouter request."})

async def call_gemini_openai_compatible(payload: dict, key: str):
    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    try:
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"]
        )
        return JSONResponse(status_code=200, content=response.model_dump())
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to complete Gemini request."})

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
    usage_data = {}
    for provider, keys in request_counts.items():
        usage_data[provider] = {
            "keys": {},
            "rate_limits": rate_limit_settings.get(provider, {})
        }
        for key, count in keys.items():
            usage_data[provider]["keys"][key] = {
                "requests": count,
                "rate_limit_windows": {
                    "req_min": len(rate_limit_windows.get(f"req_min:{provider}:{key}", [])),
                    "req_day": len(rate_limit_windows.get(f"req_day:{provider}:{key}", []))
                }
            }
    return JSONResponse(status_code=200, content=usage_data)
