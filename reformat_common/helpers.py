from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from .types import UsageDict

USAGE_KEYS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "cost",
    "upstream_inference_cost",
]


def is_na(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def single_line(value: Any) -> str:
    if value is None or is_na(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def infer_platform(url: str) -> str:
    lowered = url.lower()
    if "tiktok.com" in lowered:
        return "TikTok"
    if "facebook.com" in lowered or "fb.com" in lowered:
        return "Facebook"
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "YouTube"
    if "twitter.com" in lowered or "x.com" in lowered:
        return "X/Twitter"
    if "instagram.com" in lowered:
        return "Instagram"
    return ""


def make_client(api_key: str, base_url: str) -> OpenAI:
    extra_headers: dict[str, str] = {}
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        extra_headers["HTTP-Referer"] = referer
    if app_name := os.getenv("OPENROUTER_APP_NAME"):
        extra_headers["X-Title"] = app_name
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)


def zero_usage() -> UsageDict:
    return {key: 0.0 for key in USAGE_KEYS}


def read_usage_value(container: Any, key: str) -> float:
    if container is None:
        return 0.0
    value: Any = (
        getattr(container, key, None)
        if not isinstance(container, dict)
        else container.get(key)
    )
    return float(value) if isinstance(value, (int, float)) else 0.0


def extract_usage(response: Any) -> UsageDict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return zero_usage()

    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    completion_tokens_details = getattr(usage, "completion_tokens_details", None)
    usage_dict = {key: 0.0 for key in USAGE_KEYS}
    usage_dict["prompt_tokens"] = read_usage_value(usage, "prompt_tokens")
    usage_dict["completion_tokens"] = read_usage_value(usage, "completion_tokens")
    usage_dict["total_tokens"] = read_usage_value(usage, "total_tokens")
    usage_dict["cached_tokens"] = read_usage_value(prompt_tokens_details, "cached_tokens")
    usage_dict["cache_write_tokens"] = read_usage_value(
        prompt_tokens_details,
        "cache_write_tokens",
    )
    usage_dict["audio_tokens"] = read_usage_value(
        completion_tokens_details,
        "audio_tokens",
    )
    usage_dict["reasoning_tokens"] = read_usage_value(
        completion_tokens_details,
        "reasoning_tokens",
    )
    usage_dict["cost"] = read_usage_value(usage, "cost")
    usage_dict["upstream_inference_cost"] = read_usage_value(
        usage,
        "upstream_inference_cost",
    )
    return usage_dict


def add_usage(total_usage: UsageDict, usage: UsageDict) -> None:
    for key in USAGE_KEYS:
        total_usage[key] += usage.get(key, 0.0)


def get_temperature(model: str) -> float:
    if "gpt-5" in model.lower():
        return 1.0
    return 0.0

