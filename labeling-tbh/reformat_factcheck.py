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

# DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_MODEL = "github_copilot/gpt-5-mini"

DEFAULT_WORKERS = 16
DEFAULT_BATCH_SIZE = 15
# DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_BASE_URL = "http://localhost:4000"


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 10.0
RATE_LIMIT_MAX_DELAY = 120.0

OUTPUT_FIELDS = ["link", "video_name", "source", "ringkasan", "klaim", "fakta", "label", "analisis"]

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

ResultRow = dict[str, Any]
PARSE_LOG_LOCK = threading.Lock()

VALID_LABELS = {"UJARAN KEBENCIAN", "DISINFORMASI", "FITNAH", "NETRAL", "TIDAK DAPAT DINILAI"}

SYSTEM_PROMPT = """Kamu adalah sistem analisis konten yang mengevaluasi konten media sosial yang telah melalui proses fact-checking.

Kamu akan diberikan:
1. "Narasi" — teks/caption dari konten yang beredar
2. "Penjelasan" — analisis dari tim fact-checker (jika ada)
3. "Kesimpulan" — kesimpulan singkat fact-checker (jika ada)

Tugasmu — ekstrak dan tentukan:
1. Ringkasan netral tentang APA isi konten yang beredar.
2. Klaim utama yang dibuat dalam konten secara literal.
3. Fakta atau bukti yang menyangkal/mengonfirmasi klaim (dari penjelasan fact-checker).
4. Label kategori pelanggaran.
5. Berikan analisis singkat tentang karakteristik kasus.

Aturan:
- ringkasan : 3–5 kalimat netral tentang isi konten. Fokus pada ISI, bukan penilaian.
- klaim     : tulis klaim utama konten  kalimat factual menggunakan
              sudut pandang ketiga, contoh: "Konten menyatakan bahwa ..." atau
              "Unggahan mengklaim bahwa ...".
              Tulis secara literal — apa yang dikatakan atau ditampilkan — tanpa
              interpretasi, penilaian, atau framing evaluatif.
              Jika tidak ada klaim spesifik, tulis "tidak ada detail konten".
- fakta     : tulis FAKTA atau bukti yang menyangkal klaim — fokus pada substansinya:
              apa yang sebenarnya terjadi, data teknis, pernyataan resmi, atau
              klarifikasi dari sumber otoritatif.
              Jangan menyebut nama organisasi fact-checker (Mafindo, TurnBackHoax, dll)
              sebagai sumber utama — tulis faktanya, bukan siapa yang memeriksa.
              Jika tidak ada bukti penyangkal yang konkret, tulis "tidak ada".
- label     : pilih SATU dari LIMA label berikut:
                - Fitnah :
                  *(UU No. 1 Tahun 2024 - Pasal 27A) 
                  * Menyerang kehormatan/nama baik dengan menuduhkan hal tertentu melalui sistem elektronik agar diketahui umum.
                - Ujaran kebencian :
                  *(UU No. 1 Tahun 2024 - Pasal 28 ayat 2)
                  * Melarang penyebaran informasi yang bertujuan menimbulkan rasa kebencian atau permusuhan individu/kelompok berdasarkan SARA.
                - Disinformasi :
                  * Bedasarkan (UU No. 1 Tahun 2024):
                  * Pasal 28 ayat (1): "Setiap Orang dengan sengaja dan tanpa hak menyebarkan Berita Bohong dan menyesatkan yang mengakibatkan kerugian materiil bagi konsumen dalam Transaksi Elektronik."
                  * Pasal 28 ayat (3): "Setiap Orang dengan sengaja dan tanpa hak menyebarkan Informasi Elektronik dan/atau Dokumen Elektronik yang diketahui memuat berita bohong atau menyesatkan yang mengakibatkan kerusuhan di masyarakat." 
                - NETRAL : konten yang tidak melanggar
                - TIDAK DAPAT DINILAI : tidak cukup informasi untuk menilai
- analisis  : jelaskan kenapa konten ini bermasalah — fokus pada apa yang diklaim, apakah ada
              bukti yang mendukung atau menyangkal, serta dampak potensial yang dapat ditimbulkan.
              Jangan sebut nama kategori pelanggarannya secara eksplisit.
              Sertakan kutipan atau konten spesifik sebagai dasar penilaian, dan gunakan fakta jika tersedia.
              Tulis dalam 3–4 kalimat secara ringkas, tidak terlalu panjang maupun terlalu singkat.
"""


USER_PROMPT_TEMPLATE = (
    "Analisis konten berikut dan tentukan ringkasan, klaim, fakta, label, dan analisis.\n\n"
    "{content}\n\n"
    "PENTING: Kembalikan output dalam JSON object valid saja dengan tepat 5 key berikut: "
    '"ringkasan", "klaim", "fakta", "label", "analisis". '
    "Jangan tambahkan teks lain di luar JSON."
)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "analisis_factcheck",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ringkasan": {"type": "string"},
                "klaim": {"type": "string"},
                "fakta": {"type": "string"},
                "label": {"type": "string"},
                "analisis": {"type": "string"},
            },
            "required": ["ringkasan", "klaim", "fakta", "label", "analisis"],
            "additionalProperties": False,
        },
    },
}

JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}
REQUIRED_RESPONSE_KEYS = ["ringkasan", "klaim", "fakta", "label", "analisis"]


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
    if "facebook.com" in url or "fb.com" in url:
        return "Facebook"
    if "tiktok.com" in url:
        return "TikTok"
    if "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    if "twitter.com" in url or "x.com" in url:
        return "X/Twitter"
    if "instagram.com" in url:
        return "Instagram"
    return ""


def _normalize_label(raw: str) -> str:
    cleaned = raw.strip().upper()
    return cleaned if cleaned in VALID_LABELS else "TIDAK DAPAT DINILAI"


def _zero_usage() -> dict[str, float]:
    return {k: 0.0 for k in USAGE_KEYS}


def _read_usage_value(container: Any, key: str) -> float:
    if container is None:
        return 0.0
    value: Any = None
    if isinstance(container, dict):
        value = container.get(key)
    else:
        value = getattr(container, key, None)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _extract_usage(response: Any) -> dict[str, float]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return _zero_usage()

    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    completion_details = usage.get("completion_tokens_details") if isinstance(usage, dict) else getattr(usage, "completion_tokens_details", None)
    cost_details = usage.get("cost_details") if isinstance(usage, dict) else getattr(usage, "cost_details", None)

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


def _add_usage(total_usage: dict[str, float], usage: dict[str, float]) -> None:
    for key in USAGE_KEYS:
        total_usage[key] += usage.get(key, 0.0)


def _build_llm_input(row_data: dict[str, Any]) -> str:
    parts = []
    narasi = _single_line(row_data.get("narasi", ""))
    penjelasan = _single_line(row_data.get("penjelasan", ""))
    kesimpulan = _single_line(row_data.get("kesimpulan", ""))

    if narasi:
        parts.append(f'Narasi:\n"""\n{narasi}\n"""')
    if penjelasan:
        parts.append(f'Penjelasan fact-checker:\n"""\n{penjelasan}\n"""')
    if kesimpulan:
        parts.append(f'Kesimpulan:\n"""\n{kesimpulan}\n"""')

    return "\n\n".join(parts)


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
                candidate = stripped[start: i + 1]
                return json.loads(candidate)

    raise json.JSONDecodeError("No complete JSON object found", content, 0)


def _validate_result_payload(payload: dict[str, Any]) -> dict[str, str]:
    missing = [k for k in REQUIRED_RESPONSE_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    normalized: dict[str, str] = {}
    for key in REQUIRED_RESPONSE_KEYS:
        value = payload.get(key, "")
        normalized[key] = value if isinstance(value, str) else str(value)
    return normalized


def _debug_parse_failure(response: Any, attempt: int, row_index: int | None) -> None:
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    refusal = getattr(message, "refusal", None) if message is not None else None
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    raw_content = getattr(message, "content", "") if message is not None else ""
    raw_text = raw_content if isinstance(raw_content, str) else str(raw_content)
    snippet = raw_text[:300].replace("\n", " ") if raw_text else "<EMPTY>"
    print(
        "  Debug parse"
        f" row={row_index if row_index is not None else '-'}"
        f" attempt={attempt}"
        f" finish_reason={finish_reason}"
        f" refusal={bool(refusal)}"
        f" text_len={len(raw_text)}"
        f" snippet={snippet!r}"
    )


def _append_parse_failure_log(
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
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    raw_content = getattr(message, "content", "") if message is not None else ""
    raw_text = raw_content if isinstance(raw_content, str) else str(raw_content)
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
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _get_temperature(model: str) -> float:
    if "gpt-5" in model.lower():
        return 1.0
    return 0.0


def _call_api(
    content: str,
    client: OpenAI,
    model: str,
    timeout: float,
    response_format: Any,
) -> Any:
    return client.chat.completions.create(
        model=model,
        temperature=_get_temperature(model),
        response_format=response_format,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(content=content)},
        ],
        timeout=timeout,
    )


def _parse_response(response: Any) -> tuple[dict[str, str], dict[str, float]]:
    raw_content = response.choices[0].message.content or ""
    parsed = _extract_json(raw_content)
    validated = _validate_result_payload(parsed)
    validated["label"] = _normalize_label(validated.get("label", ""))
    return validated, _extract_usage(response)


def _make_client(api_key: str, base_url: str) -> OpenAI:
    extra_headers: dict[str, str] = {}
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        extra_headers["HTTP-Referer"] = referer
    if app_name := os.getenv("OPENROUTER_APP_NAME"):
        extra_headers["X-Title"] = app_name
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)


def generate_fields(
    content: str,
    client: OpenAI,
    model: str,
    timeout: float = 60.0,
    row_index: int | None = None,
    parse_log_path: str | None = None,
) -> tuple[dict[str, str], dict[str, float]]:
    delay = RATE_LIMIT_BASE_DELAY
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        response: Any = None
        try:
            response = _call_api(content, client, model, timeout, cast(Any, RESPONSE_SCHEMA))
            return _parse_response(response)

        except (json.JSONDecodeError, ValueError) as exc:
            _debug_parse_failure(response, attempt, row_index)
            _append_parse_failure_log(parse_log_path, row_index, attempt, response, exc)
            # fallback ke json_object untuk edge-case provider/model
            try:
                response = _call_api(content, client, model, timeout, JSON_OBJECT_RESPONSE_FORMAT)
                result = _parse_response(response)
                print(f"  Fallback json_object sukses row {row_index if row_index is not None else '-'} attempt {attempt}")
                return result
            except (json.JSONDecodeError, ValueError) as fallback_exc:
                _debug_parse_failure(response, attempt, row_index)
                _append_parse_failure_log(parse_log_path, row_index, attempt, response, fallback_exc)
            if attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            print(f"  JSON/schema error row {row_index if row_index is not None else '-'}, retry ({attempt}/{RATE_LIMIT_MAX_RETRIES}): {exc}")
            time.sleep(2)

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

        except APIStatusError as exc:
            if attempt < RATE_LIMIT_MAX_RETRIES and exc.status_code >= 500:
                wait = min(delay, RATE_LIMIT_MAX_DELAY)
                print(f"  Server error {exc.status_code}, tunggu {wait:.0f}s (percobaan {attempt}/{RATE_LIMIT_MAX_RETRIES})")
                time.sleep(wait)
                delay = min(delay * 2, RATE_LIMIT_MAX_DELAY)
            else:
                raise

    raise RuntimeError("Semua percobaan habis")


def _build_result_row(
    row_data: dict[str, Any],
    generated: dict[str, Any],
) -> ResultRow:
    link = _single_line(row_data.get("link_video_asli", "")) or _single_line(row_data.get("link_article", ""))
    source = _infer_platform(link) if link else ""
    return {
        "link": link,
        "video_name": _single_line(row_data.get("video_name", "")),
        "source": source,
        "ringkasan": _single_line(generated.get("ringkasan", "")),
        "klaim": _single_line(generated.get("klaim", "")),
        "fakta": _single_line(generated.get("fakta", "tidak ada")),
        "label": _single_line(generated.get("label", "TIDAK DAPAT DINILAI")),
        "analisis": _single_line(generated.get("analisis", "")),
    }


def _process_one_row(
    index: int,
    row_data: dict[str, Any],
    client: OpenAI,
    model: str,
    timeout: float,
    parse_log_path: str | None = None,
) -> tuple[int, dict[str, str]]:
    content = _build_llm_input(row_data)

    if not content.strip():
        generated = {
            "ringkasan": "konten tidak tersedia",
            "klaim": "tidak ada detail konten",
            "fakta": "tidak ada",
            "label": "TIDAK DAPAT DINILAI",
            "analisis": "",
        }
        result = _build_result_row(row_data, generated)
        result["_input"] = ""
        result["_usage"] = _zero_usage()
        return index, result

    try:
        generated, usage = generate_fields(
            content,
            client,
            model,
            timeout,
            row_index=index,
            parse_log_path=parse_log_path,
        )
    except Exception as exc:
        print(f"  ⚠ Gagal generate baris {index}: {exc}")
        generated = {
            "ringkasan": "tidak dapat diproses",
            "klaim": "tidak dapat diproses",
            "fakta": "tidak ada",
            "label": "TIDAK DAPAT DINILAI",
            "analisis": "tidak dapat diproses",
        }
        usage = _zero_usage()

    result = _build_result_row(row_data, generated)
    result["_input"] = content
    result["_usage"] = usage
    return index, result


def _make_run_dir(base_output_dir: str) -> str:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _init_output_csv(output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=OUTPUT_FIELDS).writeheader()


def _append_csv_batch(output_path: str, rows: list[ResultRow]) -> None:
    if not rows:
        return
    with open(output_path, "a", encoding="utf-8", newline="") as f:
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

    print(f"  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")
    print(f"  TXT  : {txt_path}")


def _write_merged_csv(
    run_dir: str,
    all_indexed: list[tuple[int, ResultRow]],
    original_df: pd.DataFrame,
) -> None:
    results_by_idx = {
        idx: {k: v for k, v in row.items() if not k.startswith("_")}
        for idx, row in all_indexed
    }
    result_df = pd.DataFrame.from_dict(results_by_idx, orient="index")
    overlap_cols = [col for col in result_df.columns if col in original_df.columns]
    if overlap_cols:
        result_df = result_df.drop(columns=overlap_cols)
    merged = original_df.join(result_df)
    merged_path = os.path.join(run_dir, "merged.csv")
    merged.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"  MERGED CSV : {merged_path}")


def _write_run_info(
    run_dir: str,
    run_command: str,
    input_path: str,
    model: str,
    workers: int,
    batch_size: int,
    timeout: float,
    limit: int | None,
    total_rows: int,
    start_time: datetime,
    end_time: datetime,
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
        f.write(f"Cached tokens      : {int(usage_totals['cached_tokens'])}\n")
        f.write(f"Cache write tokens : {int(usage_totals['cache_write_tokens'])}\n")
        f.write(f"Audio tokens       : {int(usage_totals['audio_tokens'])}\n")
        f.write(f"Reasoning tokens   : {int(usage_totals['reasoning_tokens'])}\n")
        f.write(f"Cost               : {usage_totals['cost']:.6f}\n")
        f.write(f"Upstream cost      : {usage_totals['upstream_inference_cost']:.6f}\n")
    print(f"  RUN INFO   : {info_path}")


def reformat_csv(
    input_path: str,
    output_dir: str,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = 60.0,
    limit: int | None = None,
    run_command: str = "",
) -> None:
    if workers < 1:
        raise ValueError("workers harus >= 1")
    if batch_size < 1:
        raise ValueError("batch_size harus >= 1")

    df = pd.read_csv(input_path)

    required = {"narasi", "link_video_asli"}
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
        print("Tidak ada data. Output CSV kosong dibuat.")
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
                    fallback = {
                        "ringkasan": "tidak dapat diproses",
                        "klaim": "tidak dapat diproses",
                        "fakta": "tidak ada",
                        "label": "TIDAK DAPAT DINILAI",
                        "analisis": "tidak dapat diproses",
                    }
                    result = _build_result_row(row_dict, fallback)
                    result["_input"] = ""
                    result["_usage"] = _zero_usage()
                    batch_results.append((idx, result))
                    print(f"  ⚠ Gagal (outer) baris {idx}: {exc}")

        batch_results.sort(key=lambda x: x[0])
        ordered_rows: list[ResultRow] = [row for _, row in batch_results]
        for row in ordered_rows:
            row_usage = row.get("_usage")
            _add_usage(usage_totals, row_usage if isinstance(row_usage, dict) else _zero_usage())
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
        run_dir=run_dir,
        run_command=run_command,
        input_path=input_path,
        model=model,
        workers=workers,
        batch_size=batch_size,
        timeout=timeout,
        limit=limit,
        total_rows=total,
        start_time=start_time,
        end_time=end_time,
        usage_totals=usage_totals,
    )
    print(f"\nSelesai. Run folder: {run_dir}")


def retry_failed_run(
    run_dir: str,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = 60.0,
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
                    fallback = {"ringkasan": "tidak dapat diproses", "klaim": "tidak dapat diproses",
                                "fakta": "tidak ada", "label": "TIDAK DAPAT DINILAI", "analisis": "tidak dapat diproses"}
                    result = _build_result_row(row_dict, fallback)
                    result["_input"] = ""
                    result["_usage"] = _zero_usage()
                    batch_results.append((idx, result))
                    print(f"  ⚠ Gagal (retry) baris {idx}: {exc}")

        for _, result in batch_results:
            row_usage = result.get("_usage")
            _add_usage(usage_totals, row_usage if isinstance(row_usage, dict) else _zero_usage())

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
        f.write("Usage\n")
        f.write(f"Prompt tokens      : {int(usage_totals['prompt_tokens'])}\n")
        f.write(f"Completion tokens  : {int(usage_totals['completion_tokens'])}\n")
        f.write(f"Total tokens       : {int(usage_totals['total_tokens'])}\n")
        f.write(f"Cached tokens      : {int(usage_totals['cached_tokens'])}\n")
        f.write(f"Cache write tokens : {int(usage_totals['cache_write_tokens'])}\n")
        f.write(f"Audio tokens       : {int(usage_totals['audio_tokens'])}\n")
        f.write(f"Reasoning tokens   : {int(usage_totals['reasoning_tokens'])}\n")
        f.write(f"Cost               : {usage_totals['cost']:.6f}\n")
        f.write(f"Upstream cost      : {usage_totals['upstream_inference_cost']:.6f}\n")
    print(f"  RUN INFO   : {info_path}")

    still_failed = sum(1 for r in all_rows if r.get("ringkasan") == "tidak dapat diproses")
    print(f"\nSelesai. {total_failed - still_failed}/{total_failed} baris berhasil diperbaiki.", end="")
    if still_failed:
        print(f" Masih gagal: {still_failed} baris.")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat fact-check CSV ke format ringkasan/klaim/fakta/label/analisis"
    )
    parser.add_argument("input", nargs="?", help="Path file CSV input")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT_DIR, help=f"Folder output (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--retry-failed", metavar="RUN_DIR", help="Retry baris gagal dari run sebelumnya")
    parser.add_argument("--api-key", help="API key (default: env OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Base URL API (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Jumlah thread paralel (default: {DEFAULT_WORKERS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Baris per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout dalam detik (default: 60)")
    parser.add_argument("--limit", type=int, default=None, help="Proses hanya N baris pertama (untuk testing)")
    args = parser.parse_args()

    key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        parser.error("API key belum diset (--api-key atau env OPENROUTER_API_KEY)")

    model = args.model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if args.retry_failed:
        retry_failed_run(
            run_dir=args.retry_failed,
            api_key=key,
            base_url=args.base_url,
            model=model,
            workers=args.workers,
            batch_size=args.batch_size,
            timeout=args.timeout,
            run_command=" ".join(sys.argv),
        )
    else:
        if not args.input:
            parser.error("input wajib diisi (atau gunakan --retry-failed)")
        reformat_csv(
            input_path=args.input,
            output_dir=args.output,
            api_key=key,
            base_url=args.base_url,
            model=model,
            workers=args.workers,
            batch_size=args.batch_size,
            timeout=args.timeout,
            limit=args.limit,
            run_command=" ".join(sys.argv),
        )


if __name__ == "__main__":
    main()
