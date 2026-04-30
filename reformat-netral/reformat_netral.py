import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
import sys
import threading
import time
from typing import Any, cast

import pandas as pd
from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel

DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 15
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 10.0
RATE_LIMIT_MAX_DELAY = 120.0

OUTPUT_FIELDS = ["link", "source", "ringkasan", "klaim", "fakta", "label", "analisis"]

USAGE_KEYS = [
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cached_tokens", "cache_write_tokens", "audio_tokens",
    "reasoning_tokens", "cost", "upstream_inference_cost",
]

ResultRow = dict[str, Any]
PARSE_LOG_LOCK = threading.Lock()


# ── Pydantic output schema ────────────────────────────────────────────────────

class AnalisisResult(BaseModel):
    ringkasan: str
    klaim: str
    fakta: str
    analisis: str


REQUIRED_RESPONSE_KEYS = ["ringkasan", "klaim", "fakta", "analisis"]
JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah sistem analisis konten media sosial yang membantu menyusun dataset.

Kamu akan diberikan caption/teks dari sebuah konten TikTok.

Tugasmu — ekstrak empat hal:
1. Ringkasan netral tentang APA isi konten.
2. Klaim atau pernyataan utama yang disampaikan konten.
3. Fakta atau informasi faktual yang ada di dalam teks.
4. Analisis singkat mengenai karakteristik konten yang dimuat.

Aturan:
- ringkasan : 1–2 kalimat netral tentang isi umum konten. Fokus pada gambaran besar isi.
              Jangan menyebut detail spesifik seperti angka, tanggal, nama tokoh, sumber, atau hashtag.
              Jangan evaluatif atau mengandung opini.

- klaim     : tulis inti informasi yang disampaikan konten dalam sudut pandang orang ketiga.
              Gunakan format seperti "Konten menyatakan bahwa ..." atau "Unggahan menginformasikan bahwa ...".
              Fokus pada isi utama dan cara penyampaiannya.
              Tidak boleh hanya berupa daftar kata atau entitas.
              Jika tidak ada pernyataan utama, tulis "tidak ada detail konten".

- fakta     : tulis semua informasi faktual yang disebutkan dalam teks dalam bentuk kalimat lengkap.
              Fakta mencakup peristiwa, pelaku, waktu, lokasi, angka, dan deskripsi kejadian.
              Setiap fakta harus ditulis sebagai kalimat utuh, bukan daftar kata atau keyword.
              Jangan memasukkan hashtag, sumber media, link, atau credit (misalnya CNN, REUTERS).
              Jika tidak ada informasi faktual, tulis "tidak ada".

- analisis  : Jelaskan karakteristik konten dalam 3–4 kalimat padat dengan struktur:
              [Identifikasi Jenis Konten] -> [Alasan] -> [Bukti Kutipan Spesifik].
              Fokus pada pola konten, bukan penilaian benar/salah.
              Tidak perlu menyebut label struktur secara eksplisit.
              Harus objektif, ringkas, dan konsisten.
"""

USER_PROMPT_TEMPLATE = (
    "Analisis caption konten TikTok berikut.\n\n"
    "Caption:\n\"\"\"\n{caption}\n\"\"\"\n\n"
    "PENTING: Kembalikan output dalam JSON object valid saja dengan tepat 4 key berikut: "
    '"ringkasan", "klaim", "fakta", "analisis". '
    "Jangan tambahkan teks lain di luar JSON."
)

# ─────────────────────────────────────────────────────────────────────────────


def _is_na(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _single_line(value: Any) -> str:
    if value is None or _is_na(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _infer_platform(url: str) -> str:
    url = url.lower()
    if "tiktok.com" in url:
        return "TikTok"
    if "facebook.com" in url or "fb.com" in url:
        return "Facebook"
    if "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    if "twitter.com" in url or "x.com" in url:
        return "X/Twitter"
    if "instagram.com" in url:
        return "Instagram"
    return ""


def _get_temperature(model: str) -> float:
    if "gpt-5" in model.lower():
        return 1.0
    return 0.0


def _zero_usage() -> dict[str, float]:
    return {k: 0.0 for k in USAGE_KEYS}


def _read_usage_value(container: Any, key: str) -> float:
    if container is None:
        return 0.0
    value: Any = getattr(container, key, None) if not isinstance(container, dict) else container.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _extract_usage(response: Any) -> dict[str, float]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return _zero_usage()
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cost_details = getattr(usage, "cost_details", None)
    return {
        "prompt_tokens": _read_usage_value(usage, "prompt_tokens"),
        "completion_tokens": _read_usage_value(usage, "completion_tokens"),
        "total_tokens": _read_usage_value(usage, "total_tokens"),
        "cached_tokens": _read_usage_value(prompt_details, "cached_tokens"),
        "cache_write_tokens": _read_usage_value(prompt_details, "cache_write_tokens"),
        "audio_tokens": _read_usage_value(prompt_details, "audio_tokens"),
        "reasoning_tokens": _read_usage_value(completion_details, "reasoning_tokens"),
        "cost": _read_usage_value(usage, "cost"),
        "upstream_inference_cost": _read_usage_value(cost_details, "upstream_inference_cost"),
    }


def _add_usage(total: dict[str, float], usage: dict[str, float]) -> None:
    for key in USAGE_KEYS:
        total[key] += usage.get(key, 0.0)


def _make_client(api_key: str, base_url: str) -> OpenAI:
    extra_headers: dict[str, str] = {}
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        extra_headers["HTTP-Referer"] = referer
    if app_name := os.getenv("OPENROUTER_APP_NAME"):
        extra_headers["X-Title"] = app_name
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)


# ── Parse failure logging ─────────────────────────────────────────────────────

def _debug_parse_failure(response: Any, attempt: int, row_index: int | None) -> None:
    if response is None:
        return
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    raw_content = getattr(message, "content", "") if message is not None else ""
    raw_text = raw_content if isinstance(raw_content, str) else str(raw_content)
    snippet = raw_text[:300].replace("\n", " ") if raw_text else "<EMPTY>"
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    print(
        f"  Debug parse row={row_index if row_index is not None else '-'}"
        f" attempt={attempt} finish_reason={finish_reason}"
        f" text_len={len(raw_text)} snippet={snippet!r}"
    )


def _append_parse_failure_log(
    log_path: str | None, row_index: int | None, attempt: int, response: Any, error: Exception,
) -> None:
    if not log_path or response is None:
        return
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    raw_content = getattr(message, "content", "") if message is not None else ""
    raw_text = raw_content if isinstance(raw_content, str) else str(raw_content)
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "row": row_index, "attempt": attempt,
        "finish_reason": getattr(choices[0], "finish_reason", None) if choices else None,
        "error": str(error), "text_len": len(raw_text), "raw_text": raw_text,
    }
    with PARSE_LOG_LOCK:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ── API call & parse ──────────────────────────────────────────────────────────

def _call_parse(
    caption: str, client: OpenAI, model: str, timeout: float,
) -> tuple[AnalisisResult, dict[str, float]]:
    """Call API using Pydantic structured output (.parse)."""
    response = client.beta.chat.completions.parse(
        model=model,
        temperature=_get_temperature(model),
        response_format=AnalisisResult,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(caption=caption)},
        ],
        timeout=timeout,
    )
    msg = response.choices[0].message
    if msg.refusal:
        raise ValueError(f"Model refusal: {msg.refusal}")
    if msg.parsed is None:
        raise ValueError("Parsed result is None")
    return msg.parsed, _extract_usage(response)


def _extract_json(content: str) -> dict:
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
    for i, ch in enumerate(stripped[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start: i + 1])
    raise json.JSONDecodeError("No complete JSON object found", content, 0)


def _call_json_object(
    caption: str, client: OpenAI, model: str, timeout: float,
) -> tuple[AnalisisResult, dict[str, float], Any]:
    """Fallback: json_object format with manual parse."""
    response = client.chat.completions.create(
        model=model,
        temperature=_get_temperature(model),
        response_format=JSON_OBJECT_RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(caption=caption)},
        ],
        timeout=timeout,
    )
    raw = response.choices[0].message.content or ""
    data = _extract_json(raw)
    missing = [k for k in REQUIRED_RESPONSE_KEYS if k not in data]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    parsed = AnalisisResult(
        ringkasan=str(data.get("ringkasan", "")),
        klaim=str(data.get("klaim", "")),
        fakta=str(data.get("fakta", "tidak ada")),
        analisis=str(data.get("analisis", "")),
    )
    return parsed, _extract_usage(response), response


def generate_fields(
    caption: str,
    client: OpenAI,
    model: str,
    timeout: float = 60.0,
    row_index: int | None = None,
    parse_log_path: str | None = None,
) -> tuple[AnalisisResult, dict[str, float]]:
    delay = RATE_LIMIT_BASE_DELAY
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        response: Any = None
        try:
            return _call_parse(caption, client, model, timeout)

        except (json.JSONDecodeError, ValueError) as exc:
            _debug_parse_failure(response, attempt, row_index)
            _append_parse_failure_log(parse_log_path, row_index, attempt, response, exc)
            try:
                parsed, usage, response = _call_json_object(caption, client, model, timeout)
                print(f"  Fallback json_object sukses row {row_index if row_index is not None else '-'} attempt {attempt}")
                return parsed, usage
            except (json.JSONDecodeError, ValueError) as fallback_exc:
                _debug_parse_failure(response, attempt, row_index)
                _append_parse_failure_log(parse_log_path, row_index, attempt, response, fallback_exc)
            if attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            print(f"  JSON/schema error row {row_index if row_index is not None else '-'}, retry ({attempt}/{RATE_LIMIT_MAX_RETRIES}): {exc}")
            time.sleep(2)

        except APIStatusError as exc:
            # json_schema not supported (4xx) → fallback to json_object
            if exc.status_code < 500:
                try:
                    parsed, usage, response = _call_json_object(caption, client, model, timeout)
                    print(f"  Fallback json_object sukses row {row_index if row_index is not None else '-'} attempt {attempt} (schema unsupported)")
                    return parsed, usage
                except (json.JSONDecodeError, ValueError) as fallback_exc:
                    _debug_parse_failure(response, attempt, row_index)
                    _append_parse_failure_log(parse_log_path, row_index, attempt, response, fallback_exc)
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    raise
                time.sleep(2)
            elif attempt < RATE_LIMIT_MAX_RETRIES:
                wait = min(delay, RATE_LIMIT_MAX_DELAY)
                print(f"  Server error {exc.status_code}, tunggu {wait:.0f}s (percobaan {attempt}/{RATE_LIMIT_MAX_RETRIES})")
                time.sleep(wait)
                delay = min(delay * 2, RATE_LIMIT_MAX_DELAY)
            else:
                raise

        except RateLimitError as exc:
            if attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            retry_after: float | None = None
            if hasattr(exc, "response") and exc.response is not None:
                raw = exc.response.headers.get("Retry-After")
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        pass
            wait = min(retry_after or delay, RATE_LIMIT_MAX_DELAY)
            print(f"  Rate limit, tunggu {wait:.0f}s (percobaan {attempt}/{RATE_LIMIT_MAX_RETRIES})")
            time.sleep(wait)
            delay = min(delay * 2, RATE_LIMIT_MAX_DELAY)

    raise RuntimeError("Semua percobaan habis")


# ── Row processing ────────────────────────────────────────────────────────────

def _build_result_row(row_data: dict[str, Any], parsed: AnalisisResult) -> ResultRow:
    url = _single_line(row_data.get("url", ""))
    return {
        "link": url,
        "source": _infer_platform(url) if url else _single_line(row_data.get("creator", "")),
        "ringkasan": _single_line(parsed.ringkasan),
        "klaim": _single_line(parsed.klaim),
        "fakta": _single_line(parsed.fakta) or "tidak ada",
        "label": "NETRAL",
        "analisis": _single_line(parsed.analisis),
    }


def _process_one_row(
    index: int,
    row_data: dict[str, Any],
    client: OpenAI,
    model: str,
    timeout: float,
    parse_log_path: str | None = None,
) -> tuple[int, ResultRow]:
    caption = _single_line(row_data.get("caption", ""))

    if not caption:
        parsed = AnalisisResult(
            ringkasan="caption tidak tersedia",
            klaim="tidak ada detail konten",
            fakta="tidak ada",
            analisis="",
        )
        result = _build_result_row(row_data, parsed)
        result["_input"] = ""
        result["_usage"] = _zero_usage()
        return index, result

    try:
        parsed, usage = generate_fields(caption, client, model, timeout, row_index=index, parse_log_path=parse_log_path)
    except Exception as exc:
        print(f"  ⚠ Gagal generate baris {index}: {exc}")
        parsed = AnalisisResult(
            ringkasan="tidak dapat diproses",
            klaim="tidak dapat diproses",
            fakta="tidak ada",
            analisis="tidak dapat diproses",
        )
        usage = _zero_usage()

    result = _build_result_row(row_data, parsed)
    result["_input"] = caption
    result["_usage"] = usage
    return index, result


# ── Output helpers ────────────────────────────────────────────────────────────

def _make_run_dir(base: str) -> str:
    run_dir = os.path.join(base, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _init_output_csv(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=OUTPUT_FIELDS).writeheader()


def _append_csv_batch(path: str, rows: list[ResultRow]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def _write_debug_outputs(run_dir: str, csv_path: str, all_rows: list[ResultRow]) -> None:
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(run_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(all_rows, 1):
            f.write(f"=== [{i}] ===\n")
            f.write(f"Link   : {row.get('link', '')}\n")
            f.write(f"Source : {row.get('source', '')}\n")
            f.write(f"Label  : {row.get('label', '')}\n\n")
            f.write(f"[INPUT ke LLM]:\n{row.get('_input', '')}\n\n")
            f.write(f"Ringkasan:\n{row.get('ringkasan', '')}\n\n")
            f.write(f"Klaim:\n{row.get('klaim', '')}\n\n")
            f.write(f"Fakta:\n{row.get('fakta', '')}\n\n")
            f.write(f"Analisis:\n{row.get('analisis', '')}\n")
            f.write("\n======\n\n")

    print(f"  CSV        : {csv_path}")
    print(f"  JSON       : {json_path}")
    print(f"  TXT        : {txt_path}")


def _write_merged_csv(run_dir: str, all_indexed: list[tuple[int, ResultRow]], original_df: pd.DataFrame) -> None:
    results_by_idx = {
        idx: {k: v for k, v in row.items() if not k.startswith("_")}
        for idx, row in all_indexed
    }
    result_df = pd.DataFrame.from_dict(results_by_idx, orient="index")
    overlap = [c for c in result_df.columns if c in original_df.columns]
    if overlap:
        result_df = result_df.drop(columns=overlap)
    merged = original_df.join(result_df)
    merged_path = os.path.join(run_dir, "merged.csv")
    merged.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"  MERGED CSV : {merged_path}")


def _write_run_info(
    run_dir: str, run_command: str, input_path: str, model: str,
    workers: int, batch_size: int, timeout: float, limit: int | None,
    total_rows: int, start_time: datetime, end_time: datetime,
    usage_totals: dict[str, float],
) -> None:
    duration = (end_time - start_time).total_seconds()
    info_path = os.path.join(run_dir, "run_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Run Info\n========\n")
        f.write(f"Date       : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command    : {run_command}\n")
        f.write(f"Input      : {os.path.abspath(input_path)}\n")
        f.write(f"Output dir : {run_dir}\n")
        f.write(f"Model      : {model}\n")
        f.write(f"Workers    : {workers}\n")
        f.write(f"Batch size : {batch_size}\n")
        f.write(f"Timeout    : {timeout}s\n")
        f.write(f"Limit      : {limit if limit else 'none (all rows)'}\n")
        f.write(f"Rows       : {total_rows}\n")
        f.write(f"Duration   : {duration:.1f}s\n")
        f.write("\nUsage\n=====\n")
        f.write(f"Prompt tokens      : {int(usage_totals['prompt_tokens'])}\n")
        f.write(f"Completion tokens  : {int(usage_totals['completion_tokens'])}\n")
        f.write(f"Total tokens       : {int(usage_totals['total_tokens'])}\n")
        f.write(f"Cost               : {usage_totals['cost']:.6f}\n")
        f.write(f"Upstream cost      : {usage_totals['upstream_inference_cost']:.6f}\n")
    print(f"  RUN INFO   : {info_path}")


# ── Main processing ───────────────────────────────────────────────────────────

def reformat_csv(
    input_path: str, output_dir: str, api_key: str, model: str,
    base_url: str = DEFAULT_BASE_URL, workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE, timeout: float = 60.0,
    limit: int | None = None, run_command: str = "",
) -> None:
    df = pd.read_csv(input_path)

    required = {"caption", "url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di CSV: {missing}")

    client = _make_client(api_key, base_url)
    filtered_df = df.head(limit).copy() if limit else df.copy()
    total = len(filtered_df)

    run_dir = _make_run_dir(output_dir)
    csv_path = os.path.join(run_dir, "results.csv")
    parse_log_path = os.path.join(run_dir, "parse_failures.jsonl")
    print(f"Run folder: {run_dir}")

    _init_output_csv(csv_path)
    if total == 0:
        print("Tidak ada data.")
        return

    if limit:
        print(f"Mode testing: hanya memproses {total} baris pertama.")

    start_time = datetime.now()
    done_count = 0
    usage_totals = _zero_usage()
    all_rows: list[ResultRow] = []
    all_indexed: list[tuple[int, ResultRow]] = []

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_df = filtered_df.iloc[batch_start:batch_end]
        batch_results: list[tuple[int, ResultRow]] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one_row, i, row.to_dict(), client, model, timeout, parse_log_path): i
                for i, row in batch_df.iterrows()
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_results.append(future.result())
                except Exception as exc:
                    row_dict = batch_df.loc[idx].to_dict()
                    fallback = AnalisisResult(
                        ringkasan="tidak dapat diproses",
                        klaim="tidak dapat diproses",
                        fakta="tidak ada",
                        analisis="tidak dapat diproses",
                    )
                    result = _build_result_row(row_dict, fallback)
                    result["_input"] = ""
                    result["_usage"] = _zero_usage()
                    batch_results.append((idx, result))
                    print(f"  ⚠ Gagal (outer) baris {idx}: {exc}")

        batch_results.sort(key=lambda x: x[0])
        ordered_rows = [row for _, row in batch_results]
        for row in ordered_rows:
            _add_usage(usage_totals, row.get("_usage") or _zero_usage())
        _append_csv_batch(csv_path, ordered_rows)
        all_rows.extend(ordered_rows)
        all_indexed.extend(batch_results)
        done_count += len(batch_results)
        print(f"[{done_count}/{total}] Batch selesai ({batch_start + 1}–{batch_end})")

    end_time = datetime.now()
    print("\nOutput files:")
    _write_debug_outputs(run_dir, csv_path, all_rows)
    _write_merged_csv(run_dir, all_indexed, filtered_df)
    _write_run_info(
        run_dir=run_dir, run_command=run_command, input_path=input_path,
        model=model, workers=workers, batch_size=batch_size, timeout=timeout,
        limit=limit, total_rows=total, start_time=start_time, end_time=end_time,
        usage_totals=usage_totals,
    )
    print(f"\nSelesai. Run folder: {run_dir}")


def retry_failed_run(
    run_dir: str, api_key: str, model: str,
    base_url: str = DEFAULT_BASE_URL, workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE, timeout: float = 60.0,
    run_command: str = "",
) -> None:
    merged_path = os.path.join(run_dir, "merged.csv")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"merged.csv tidak ditemukan di: {run_dir}")

    merged_df = pd.read_csv(merged_path, dtype=str).fillna("")
    failed_mask = merged_df["ringkasan"] == "tidak dapat diproses"
    failed_df = merged_df[failed_mask]
    total_failed = len(failed_df)

    if total_failed == 0:
        print("Tidak ada baris gagal. Selesai.")
        return

    print(f"Retry {total_failed} baris gagal dari {len(merged_df)} total.")
    print(f"Run folder: {run_dir}")

    client = _make_client(api_key, base_url)
    start_time = datetime.now()
    done_count = 0
    usage_totals = _zero_usage()

    for batch_start in range(0, total_failed, batch_size):
        batch_end = min(batch_start + batch_size, total_failed)
        batch_df = failed_df.iloc[batch_start:batch_end]
        batch_results: list[tuple[int, ResultRow]] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            parse_log_path = os.path.join(run_dir, "parse_failures.jsonl")
            futures = {
                pool.submit(_process_one_row, i, row.to_dict(), client, model, timeout, parse_log_path): i
                for i, row in batch_df.iterrows()
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_results.append(future.result())
                except Exception as exc:
                    row_dict = batch_df.loc[idx].to_dict()
                    fallback = AnalisisResult(
                        ringkasan="tidak dapat diproses",
                        klaim="tidak dapat diproses",
                        fakta="tidak ada",
                        analisis="tidak dapat diproses",
                    )
                    result = _build_result_row(row_dict, fallback)
                    result["_input"] = ""
                    result["_usage"] = _zero_usage()
                    batch_results.append((idx, result))
                    print(f"  ⚠ Gagal (retry) baris {idx}: {exc}")

        for _, result in batch_results:
            _add_usage(usage_totals, result.get("_usage") or _zero_usage())
        for idx, new_result in batch_results:
            for field in OUTPUT_FIELDS:
                merged_df.at[idx, field] = new_result.get(field, "")

        done_count += len(batch_results)
        print(f"[{done_count}/{total_failed}] Retry batch selesai ({batch_start + 1}–{batch_end})")

    all_rows = [
        {field: str(merged_df.at[i, field]) if field in merged_df.columns else "" for field in OUTPUT_FIELDS}
        for i in merged_df.index
    ]

    end_time = datetime.now()
    csv_path = os.path.join(run_dir, "results.csv")

    print("\nMenulis ulang output files:")
    _init_output_csv(csv_path)
    _append_csv_batch(csv_path, all_rows)
    merged_df.to_csv(merged_path, index=False, encoding="utf-8")
    _write_debug_outputs(run_dir, csv_path, all_rows)

    info_path = os.path.join(run_dir, "run_info.txt")
    duration = (end_time - start_time).total_seconds()
    with open(info_path, "a", encoding="utf-8") as f:
        f.write(f"\nRetry\n=====\n")
        f.write(f"Date     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command  : {run_command}\n")
        f.write(f"Retried  : {total_failed} baris\n")
        f.write(f"Duration : {duration:.1f}s\n")
        f.write(f"Cost     : {usage_totals['cost']:.6f}\n")
    print(f"  RUN INFO   : {info_path}")

    still_failed = sum(1 for r in all_rows if r.get("ringkasan") == "tidak dapat diproses")
    print(f"\nSelesai. {total_failed - still_failed}/{total_failed} baris berhasil diperbaiki.", end="")
    if still_failed:
        print(f" Masih gagal: {still_failed} baris.")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat netral CSV ke format ringkasan/klaim/fakta/label/analisis"
    )
    parser.add_argument("input", nargs="?", help="Path file CSV input")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT_DIR, help=f"Folder output (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--retry-failed", metavar="RUN_DIR", help="Retry baris gagal dari run sebelumnya")
    parser.add_argument("--api-key", help="API key (default: env OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        parser.error("API key belum diset (--api-key atau env OPENROUTER_API_KEY)")

    model = args.model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if args.retry_failed:
        retry_failed_run(
            run_dir=args.retry_failed, api_key=key, base_url=args.base_url,
            model=model, workers=args.workers, batch_size=args.batch_size,
            timeout=args.timeout, run_command=" ".join(sys.argv),
        )
    else:
        if not args.input:
            parser.error("input wajib diisi (atau gunakan --retry-failed)")
        reformat_csv(
            input_path=args.input, output_dir=args.output, api_key=key,
            base_url=args.base_url, model=model, workers=args.workers,
            batch_size=args.batch_size, timeout=args.timeout,
            limit=args.limit, run_command=" ".join(sys.argv),
        )


if __name__ == "__main__":
    main()
