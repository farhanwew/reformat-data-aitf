from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import time
from typing import Any

import pandas as pd
from openai import APIStatusError, OpenAI, RateLimitError

from .helpers import (
    add_usage,
    extract_usage,
    get_temperature,
    make_client,
    zero_usage,
)
from .io import (
    append_csv_batch,
    append_retry_info,
    init_output_csv,
    make_run_dir,
    write_debug_outputs,
    write_merged_csv,
    write_run_info,
)
from .parsing import (
    append_parse_failure_log,
    debug_parse_failure,
    extract_json,
    validate_result_payload,
)
from .types import GeneratedT, GenerationConfig, PreparedRow, ResultRow, RunnerConfig

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 10.0
RATE_LIMIT_MAX_DELAY = 120.0


def reformat_csv(
    config: RunnerConfig[GeneratedT],
    input_path: str,
    output_dir: str,
    api_key: str,
    model: str,
    base_url: str,
    workers: int,
    batch_size: int,
    timeout: float,
    limit: int | None,
    run_command: str,
) -> None:
    _validate_runtime(workers, batch_size)

    df = pd.read_csv(input_path)
    _validate_dataframe(df, config)

    date_range = _compute_date_range(df, config.date_range_field)
    client = make_client(api_key, base_url)
    filtered_df = df.head(limit).copy() if limit else df.copy()
    total = len(filtered_df)

    run_dir = make_run_dir(output_dir, config.run_dir_suffix)
    csv_path = os.path.join(run_dir, "results.csv")
    parse_log_path = os.path.join(run_dir, "parse_failures.jsonl")
    print(f"Run folder: {run_dir}")

    init_output_csv(csv_path, config.output_fields)
    if total == 0:
        print(config.empty_run_message)
        return

    if limit:
        print(f"Mode testing: hanya memproses {total} baris pertama.")

    start_time = datetime.now()
    summary = _run_batches(
        config=config,
        data_frame=filtered_df,
        client=client,
        model=model,
        timeout=timeout,
        batch_size=batch_size,
        workers=workers,
        parse_log_path=parse_log_path,
        csv_path=csv_path,
        batch_label="Batch selesai",
        error_label="outer",
    )
    end_time = datetime.now()

    print("\nOutput files:")
    write_debug_outputs(run_dir, csv_path, summary["all_rows"], label_field=config.label_field)
    write_merged_csv(run_dir, summary["all_indexed"], filtered_df)
    write_run_info(
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
        usage_totals=summary["usage_totals"],
        usage_keys=config.usage_report_keys,
        all_rows=summary["all_rows"],
        label_field=config.label_field,
        date_range=date_range,
    )
    print(f"\nSelesai. Run folder: {run_dir}")


def retry_failed_run(
    config: RunnerConfig[GeneratedT],
    run_dir: str,
    api_key: str,
    model: str,
    base_url: str,
    workers: int,
    batch_size: int,
    timeout: float,
    run_command: str,
) -> None:
    _validate_runtime(workers, batch_size)

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

    client = make_client(api_key, base_url)
    start_time = datetime.now()
    parse_log_path = os.path.join(run_dir, "parse_failures.jsonl")
    summary = _run_batches(
        config=config,
        data_frame=failed_df,
        client=client,
        model=model,
        timeout=timeout,
        batch_size=batch_size,
        workers=workers,
        parse_log_path=parse_log_path,
        csv_path=None,
        batch_label="Retry batch selesai",
        error_label="retry",
    )

    for index, new_result in summary["all_indexed"]:
        for field in config.output_fields:
            merged_df.at[index, field] = new_result.get(field, "")

    all_rows = [
        {
            field: str(merged_df.at[index, field]) if field in merged_df.columns else ""
            for field in config.output_fields
        }
        for index in merged_df.index
    ]

    end_time = datetime.now()
    csv_path = os.path.join(run_dir, "results.csv")
    print("\nMenulis ulang output files:")
    init_output_csv(csv_path, config.output_fields)
    append_csv_batch(csv_path, all_rows, config.output_fields)
    merged_df.to_csv(merged_path, index=False, encoding="utf-8")
    write_debug_outputs(run_dir, csv_path, all_rows, label_field=config.label_field)
    append_retry_info(
        run_dir=run_dir,
        run_command=run_command,
        total_failed=total_failed,
        start_time=start_time,
        end_time=end_time,
        usage_totals=summary["usage_totals"],
        usage_keys=config.retry_usage_report_keys,
        all_rows=all_rows,
        label_field=config.label_field,
    )

    still_failed = sum(1 for row in all_rows if row.get("ringkasan") == "tidak dapat diproses")
    print(f"\nSelesai. {total_failed - still_failed}/{total_failed} baris berhasil diperbaiki.", end="")
    if still_failed:
        print(f" Masih gagal: {still_failed} baris.")
    else:
        print()


def generate_fields(
    config: GenerationConfig[GeneratedT],
    content: str,
    client: OpenAI,
    model: str,
    timeout: float = 60.0,
    row_index: int | None = None,
    parse_log_path: str | None = None,
) -> tuple[GeneratedT, dict[str, float]]:
    delay = RATE_LIMIT_BASE_DELAY
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        response: Any = None
        try:
            result, usage, _ = _call_primary(config, content, client, model, timeout)
            return result, usage

        except (json.JSONDecodeError, ValueError) as exc:
            debug_parse_failure(response, attempt, row_index)
            append_parse_failure_log(parse_log_path, row_index, attempt, response, exc)
            try:
                result, usage, response = _call_json_object_fallback(
                    config,
                    content,
                    client,
                    model,
                    timeout,
                )
                print(
                    "  Fallback json_object sukses row"
                    f" {row_index if row_index is not None else '-'} attempt {attempt}"
                )
                return result, usage
            except (json.JSONDecodeError, ValueError) as fallback_exc:
                debug_parse_failure(response, attempt, row_index)
                append_parse_failure_log(
                    parse_log_path,
                    row_index,
                    attempt,
                    response,
                    fallback_exc,
                )
            if attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            print(
                "  JSON/schema error row"
                f" {row_index if row_index is not None else '-'},"
                f" retry ({attempt}/{RATE_LIMIT_MAX_RETRIES}): {exc}"
            )
            time.sleep(2)

        except APIStatusError as exc:
            if config.fallback_on_non_server_api_error and exc.status_code < 500:
                try:
                    result, usage, response = _call_json_object_fallback(
                        config,
                        content,
                        client,
                        model,
                        timeout,
                    )
                    print(
                        "  Fallback json_object sukses row"
                        f" {row_index if row_index is not None else '-'}"
                        f" attempt {attempt} (schema unsupported)"
                    )
                    return result, usage
                except (json.JSONDecodeError, ValueError) as fallback_exc:
                    debug_parse_failure(response, attempt, row_index)
                    append_parse_failure_log(
                        parse_log_path,
                        row_index,
                        attempt,
                        response,
                        fallback_exc,
                    )
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    raise
                time.sleep(2)
            elif attempt < RATE_LIMIT_MAX_RETRIES and exc.status_code >= 500:
                wait = min(delay, RATE_LIMIT_MAX_DELAY)
                print(
                    f"  Server error {exc.status_code},"
                    f" tunggu {wait:.0f}s (percobaan {attempt}/{RATE_LIMIT_MAX_RETRIES})"
                )
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
            print(
                f"  Rate limit, tunggu {wait:.0f}s"
                f" (percobaan {attempt}/{RATE_LIMIT_MAX_RETRIES})"
            )
            time.sleep(wait)
            delay = min(delay * 2, RATE_LIMIT_MAX_DELAY)

    raise RuntimeError("Semua percobaan habis")


def _compute_date_range(df: pd.DataFrame, date_field: str | None) -> tuple[str, str] | None:
    if not date_field or date_field not in df.columns:
        return None
    col = pd.to_datetime(df[date_field], errors="coerce").dropna()
    if col.empty:
        return None
    return str(col.min().date()), str(col.max().date())


def _validate_runtime(workers: int, batch_size: int) -> None:
    if workers < 1:
        raise ValueError("workers harus >= 1")
    if batch_size < 1:
        raise ValueError("batch_size harus >= 1")


def _validate_dataframe(df: pd.DataFrame, config: RunnerConfig[GeneratedT]) -> None:
    missing = config.required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di CSV: {missing}")
    if config.validate_input_df is not None:
        config.validate_input_df(df)


def _run_batches(
    config: RunnerConfig[GeneratedT],
    data_frame: pd.DataFrame,
    client: OpenAI,
    model: str,
    timeout: float,
    batch_size: int,
    workers: int,
    parse_log_path: str,
    csv_path: str | None,
    batch_label: str,
    error_label: str,
) -> dict[str, Any]:
    total = len(data_frame)
    done_count = 0
    usage_totals = zero_usage()
    all_rows: list[ResultRow] = []
    all_indexed: list[tuple[int, ResultRow]] = []

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_df = data_frame.iloc[batch_start:batch_end]
        batch_results: list[tuple[int, ResultRow]] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_row,
                    config,
                    index,
                    row.to_dict(),
                    client,
                    model,
                    timeout,
                    parse_log_path,
                ): index
                for index, row in batch_df.iterrows()
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    batch_results.append(future.result())
                except Exception as exc:
                    row_data = batch_df.loc[index].to_dict()
                    result = config.build_result_row(
                        row_data,
                        config.build_error_result(row_data),
                    )
                    result["_input"] = ""
                    result["_usage"] = zero_usage()
                    batch_results.append((index, result))
                    print(f"  ⚠ Gagal ({error_label}) baris {index}: {exc}")

        batch_results.sort(key=lambda item: item[0])
        ordered_rows = [row for _, row in batch_results]
        for row in ordered_rows:
            row_usage = row.get("_usage")
            add_usage(
                usage_totals,
                row_usage if isinstance(row_usage, dict) else zero_usage(),
            )
        if csv_path is not None:
            append_csv_batch(csv_path, ordered_rows, config.output_fields)
        all_rows.extend(ordered_rows)
        all_indexed.extend(batch_results)
        done_count += len(batch_results)
        print(f"[{done_count}/{total}] {batch_label} ({batch_start + 1}–{batch_end})")

    return {
        "usage_totals": usage_totals,
        "all_rows": all_rows,
        "all_indexed": all_indexed,
    }


def _process_one_row(
    config: RunnerConfig[GeneratedT],
    index: int,
    row_data: dict[str, Any],
    client: OpenAI,
    model: str,
    timeout: float,
    parse_log_path: str | None = None,
) -> tuple[int, ResultRow]:
    prepared = config.prepare_row(row_data)

    if prepared.skip_generation:
        generated = prepared.preset_result
        if generated is None:
            raise ValueError("Prepared row skipped generation without preset result")
        result = config.build_result_row(row_data, generated)
        result["_input"] = prepared.llm_input
        result["_usage"] = zero_usage()
        return index, result

    try:
        generated, usage = generate_fields(
            config.generation,
            prepared.llm_input,
            client,
            model,
            timeout,
            row_index=index,
            parse_log_path=parse_log_path,
        )
    except Exception as exc:
        print(f"  ⚠ Gagal generate baris {index}: {exc}")
        generated = config.build_error_result(row_data)
        usage = zero_usage()

    result = config.build_result_row(row_data, generated)
    result["_input"] = prepared.llm_input
    result["_usage"] = usage
    return index, result


def _call_primary(
    config: GenerationConfig[GeneratedT],
    content: str,
    client: OpenAI,
    model: str,
    timeout: float,
) -> tuple[GeneratedT, dict[str, float], Any]:
    messages = _build_messages(config, content)
    if config.primary_mode == "pydantic_parse":
        if config.pydantic_model is None:
            raise ValueError("pydantic_model is required for pydantic_parse mode")
        response = client.beta.chat.completions.parse(
            model=model,
            temperature=get_temperature(model),
            response_format=config.pydantic_model,
            messages=messages,
            timeout=timeout,
        )
        message = response.choices[0].message
        if message.refusal:
            raise ValueError(f"Model refusal: {message.refusal}")
        if message.parsed is None:
            raise ValueError("Parsed result is None")
        return message.parsed, extract_usage(response), response

    response = client.chat.completions.create(
        model=model,
        temperature=get_temperature(model),
        response_format=config.primary_response_format,
        messages=messages,
        timeout=timeout,
    )
    return _parse_response_payload(config, response), extract_usage(response), response


def _call_json_object_fallback(
    config: GenerationConfig[GeneratedT],
    content: str,
    client: OpenAI,
    model: str,
    timeout: float,
) -> tuple[GeneratedT, dict[str, float], Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=get_temperature(model),
        response_format=config.fallback_response_format,
        messages=_build_messages(config, content),
        timeout=timeout,
    )
    return _parse_response_payload(config, response), extract_usage(response), response


def _parse_response_payload(
    config: GenerationConfig[GeneratedT],
    response: Any,
) -> GeneratedT:
    raw_content = response.choices[0].message.content or ""
    payload = extract_json(raw_content)
    validated = validate_result_payload(payload, config.required_response_keys)
    if config.normalize_payload is not None:
        validated = config.normalize_payload(validated)
    return config.payload_to_result(validated)


def _build_messages(
    config: GenerationConfig[GeneratedT],
    content: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": config.system_prompt},
        {
            "role": "user",
            "content": config.user_prompt_template.format(
                **{config.user_prompt_var: content}
            ),
        },
    ]
