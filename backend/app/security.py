from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request


class RateLimitExceeded(RuntimeError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Too many requests.")
        self.retry_after_seconds = retry_after_seconds


def client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    client = request.client
    if client and client.host:
        return client.host
    return "unknown"


@dataclass
class SharedRateLimiter:
    store: Any
    limit: int
    window_seconds: int
    key_prefix: str = "ip"

    def check(self, key: str) -> None:
        now = time.time()
        bucket_key = f"{self.key_prefix}:{key}"
        timestamps = self.store.get(bucket_key) or []
        fresh = [stamp for stamp in timestamps if stamp > now - self.window_seconds]
        if len(fresh) >= self.limit:
            oldest = min(fresh)
            retry_after = max(1, math.ceil(self.window_seconds - (now - oldest)))
            raise RateLimitExceeded(retry_after)
        fresh.append(now)
        self.store.put(bucket_key, fresh)
