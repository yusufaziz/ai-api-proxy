# AI Proxy

A FastAPI-based proxy server that manages multiple AI provider keys with rate limiting and load balancing capabilities.

## Features

- OpenAI-compatible API endpoints
- Support for multiple providers (OpenRouter and Gemini)
- Rate limiting per key and provider
- Load balancing across multiple API keys
- Request tracking and usage statistics
- Streaming support
- Function/Tools support
- Auto-model selection based on availability

## Configuration

Create a `config.json` file in the root directory:

```json
{
  "access_key": "your-proxy-access-key",
  "model_usage_gap_percentage": 5,
  "auto-models": [
    "gpt-3.5-turbo:free",
    "gemini-pro"
  ],
  "provider_keys": [
    {
      "provider": "openrouter",
      "key": [
        "key1",
        "key2"
      ],
      "max_request_day": 1500,
      "max_token_min": 150000,
      "max_request_min": 15
    },
    {
      "provider": "gemini",
      "key": [
        "key1",
        "key2"
      ],
      "max_request_day": 1500,
      "max_token_min": 150000,
      "max_request_min": 15
    }
  ]
}
```

### Configuration Parameters

- `access_key`: Your proxy authentication key
- `model_usage_gap_percentage`: Maximum allowed usage gap between keys (for load balancing)
- `auto-models`: List of models to try when using auto-model selection
- `provider_keys`: Provider-specific configurations
  - `provider`: Provider identifier ("openrouter" or "gemini")
  - `key`: Array of API keys
  - `max_request_day`: Maximum requests per day per key
  - `max_token_min`: Maximum tokens per minute per key
  - `max_request_min`: Maximum requests per minute per key

## API Endpoints

### Chat Completion
```http
POST /v1/chat/completions
Authorization: Bearer your-proxy-access-key

{
  "model": "gpt-3.5-turbo:free",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

### List Models
```http
GET /v1/models
Authorization: Bearer your-proxy-access-key
```

### Usage Statistics
```http
GET /v1/usage
Authorization: Bearer your-proxy-access-key
```

## Features Details

### Auto-Model Selection
Use `"model": "auto-model"` to automatically select an available model from the `auto-models` list based on current usage and rate limits.

### Rate Limiting
Rate limits are tracked per key and include:
- Per-minute request limits
- Per-day request limits
- Token usage limits (per minute)

### Load Balancing
Keys are selected based on:
1. Current rate limit status
2. Usage distribution (controlled by `model_usage_gap_percentage`)
3. Minimum usage priority

### Streaming Support
Set `"stream": true` in your request to receive streaming responses in SSE format.

### Tools/Functions Support
Both providers support OpenAI-compatible function calling format. Include `tools` array in your request to use this feature.

## Usage Data Format

The `/v1/usage` endpoint returns:
```json
{
  "overview": {
    "provider_name": {
      "total_requests": 100,
      "total_capacity": 3000,
      "usage_percentage": 3.33
    }
  },
  "details": {
    "provider_name": {
      "keys": {
        "key_id": {
          "requests": 50,
          "usage_percentage": 3.33,
          "rate_limit_windows": {
            "req_min": 5,
            "req_day": 50
          }
        }
      },
      "rate_limits": {
        "max_request_day": 1500,
        "max_token_min": 150000,
        "max_request_min": 15
      }
    }
  }
}
```
