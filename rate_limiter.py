import json
import time
from typing import Dict, List
from collections import defaultdict
import os

class RateLimiter:
    def __init__(self, storage_file: str = "rate_limit_data.json"):
        self.storage_file = storage_file
        self.request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.rate_limit_windows: Dict[str, List[float]] = defaultdict(list)
        self.load_data()

    def load_data(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.request_counts = defaultdict(lambda: defaultdict(int), data.get('request_counts', {}))
                    self.token_counts = defaultdict(lambda: defaultdict(int), data.get('token_counts', {}))
                    self.rate_limit_windows = defaultdict(list, data.get('rate_limit_windows', {}))
            except json.JSONDecodeError:
                self._initialize_empty_file()
        else:
            self._initialize_empty_file()

    def _initialize_empty_file(self):
        empty_data = {
            'request_counts': {},
            'token_counts': {},
            'rate_limit_windows': {}
        }
        with open(self.storage_file, 'w') as f:
            json.dump(empty_data, f)
        
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.token_counts = defaultdict(lambda: defaultdict(int))
        self.rate_limit_windows = defaultdict(list)

    def save_data(self):
        data = {
            'request_counts': dict(self.request_counts),
            'token_counts': dict(self.token_counts),
            'rate_limit_windows': dict(self.rate_limit_windows)
        }
        with open(self.storage_file, 'w') as f:
            json.dump(data, f)

    def is_rate_limited(self, provider: str, key: str, settings: Dict[str, int]) -> bool:
        now = time.time()
        window_min = 60

        # Per-minute request limit
        req_min_window = [t for t in self.rate_limit_windows[f"req_min:{provider}:{key}"] if t > now - window_min]
        if len(req_min_window) >= settings["max_request_min"]:
            return True
        self.rate_limit_windows[f"req_min:{provider}:{key}"] = req_min_window + [now]

        # Per-day request limit
        req_day_window = [t for t in self.rate_limit_windows[f"req_day:{provider}:{key}"] if t > now - 86400]
        if len(req_day_window) >= settings["max_request_day"]:
            return True
        self.rate_limit_windows[f"req_day:{provider}:{key}"] = req_day_window + [now]

        self.request_counts[provider][key] += 1
        self.save_data()
        return False

    def get_usage_data(self, rate_limit_settings: Dict[str, Dict[str, int]]) -> dict:
        usage_data = {
            "overview": {},
            "details": {}
        }
        
        for provider, keys in self.request_counts.items():
            # Calculate provider overview
            max_requests_per_day = rate_limit_settings[provider]["max_request_day"]
            total_provider_requests = sum(keys.values())
            total_provider_capacity = max_requests_per_day * len(keys)
            usage_percentage = (total_provider_requests / total_provider_capacity) * 100 if total_provider_capacity > 0 else 0
            
            usage_data["overview"][provider] = {
                "total_requests": total_provider_requests,
                "total_capacity": total_provider_capacity,
                "usage_percentage": round(usage_percentage, 2)
            }
            
            # Detailed per-key usage
            usage_data["details"][provider] = {
                "keys": {},
                "rate_limits": rate_limit_settings.get(provider, {})
            }
            
            for key, count in keys.items():
                key_usage_percent = (count / max_requests_per_day) * 100
                usage_data["details"][provider]["keys"][key] = {
                    "requests": count,
                    "usage_percentage": round(key_usage_percent, 2),
                    "rate_limit_windows": {
                        "req_min": len(self.rate_limit_windows.get(f"req_min:{provider}:{key}", [])),
                        "req_day": len(self.rate_limit_windows.get(f"req_day:{provider}:{key}", []))
                    }
                }
                
        return usage_data
