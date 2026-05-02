from __future__ import annotations

from datetime import datetime
import json
import threading
from typing import Any

PARSE_LOG_LOCK = threading.Lock()


def extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    stripped = content.strip()
    start = stripped.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", content, 0)

    depth = 0
    in_string = False
    escape_next = False
    for index, char in enumerate(stripped[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : index + 1])

    raise json.JSONDecodeError("No complete JSON object found", content, 0)


def validate_result_payload(
    payload: dict[str, Any],
    required_keys: list[str],
) -> dict[str, str]:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    normalized: dict[str, str] = {}
    for key in required_keys:
        value = payload.get(key, "")
        normalized[key] = value if isinstance(value, str) else str(value)
    return normalized


def debug_parse_failure(response: Any, attempt: int, row_index: int | None) -> None:
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    refusal = getattr(message, "refusal", None) if message is not None else None
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    raw_text = getattr(message, "content", "") if message is not None else ""
    print(
        "  Parse failure"
        f" row={row_index if row_index is not None else '-'}"
        f" attempt={attempt}"
        f" finish_reason={finish_reason}"
        f" refusal={bool(refusal)}"
        f" text_len={len(raw_text)}"
    )


def append_parse_failure_log(
    log_path: str | None,
    row_index: int | None,
    attempt: int,
    response: Any,
    error: Exception,
) -> None:
    if not log_path:
        return

    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    refusal = getattr(message, "refusal", None) if message is not None else None
    raw_text = getattr(message, "content", "") if message is not None else ""
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "row": row_index,
        "attempt": attempt,
        "finish_reason": finish_reason,
        "refusal": bool(refusal),
        "error": str(error),
        "text_len": len(raw_text),
        "raw_text": raw_text,
    }
    with PARSE_LOG_LOCK:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

