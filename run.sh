#!/bin/bash

# Activate virtualenv if needed
# source venv/bin/activate

echo "Running FastAPI Proxy on http://localhost:8000"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
