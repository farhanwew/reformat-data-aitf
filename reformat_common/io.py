from __future__ import annotations

import csv
from datetime import datetime
import json
import os
import re

import pandas as pd

from .types import ResultRow, UsageDict


def _redact_api_key(command: str) -> str:
    return re.sub(r"(--api-key\s+)\S+", r"\1***", command)


def make_run_dir(base_output_dir: str, suffix: str) -> str:
    run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    cleaned_suffix = suffix.strip().lower()
    if cleaned_suffix:
        run_name = f"{run_name}-{cleaned_suffix}"
    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def init_output_csv(output_path: str, output_fields: list[str]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=output_fields).writeheader()


def append_csv_batch(
    output_path: str,
    rows: list[ResultRow],
    output_fields: list[str],
) -> None:
    if not rows:
        return
    with open(output_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in output_fields})


def write_debug_outputs(
    run_dir: str,
    csv_path: str,
    all_rows: list[ResultRow],
    label_field: str | None = None,
) -> None:
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, ensure_ascii=False, indent=2)

    txt_path = os.path.join(run_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        for index, row in enumerate(all_rows, 1):
            handle.write(f"=== [{index}] ===\n")
            handle.write(f"Link   : {row.get('link', '')}\n")
            handle.write(f"Source : {row.get('source', '')}\n")
            handle.write(f"Label  : {row.get('label', '')}\n\n")
            handle.write(f"[INPUT ke LLM]:\n{row.get('_input', '')}\n\n")
            handle.write(f"Ringkasan:\n{row.get('ringkasan', '')}\n\n")
            handle.write(f"Klaim:\n{row.get('klaim', '')}\n\n")
            handle.write(f"Fakta:\n{row.get('fakta', '')}\n\n")
            handle.write(f"Analisis:\n{row.get('analisis', '')}\n")
            handle.write("\n======\n\n")

        if label_field:
            counts: dict[str, int] = {}
            for row in all_rows:
                label = str(row.get(label_field, "") or "").strip() or "(kosong)"
                counts[label] = counts.get(label, 0) + 1
            handle.write("==============================\n")
            handle.write("DISTRIBUSI LABEL\n")
            handle.write("==============================\n")
            total = len(all_rows)
            for label, count in sorted(counts.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total else 0
                handle.write(f"  {label:<30} {count:>4}  ({pct:.1f}%)\n")
            handle.write(f"  {'TOTAL':<30} {total:>4}\n")
            handle.write("==============================\n")

    print(f"  CSV        : {csv_path}")
    print(f"  JSON       : {json_path}")
    print(f"  TXT        : {txt_path}")


def write_merged_csv(
    run_dir: str,
    all_indexed: list[tuple[int, ResultRow]],
    original_df: pd.DataFrame,
) -> str:
    results_by_idx = {
        index: {key: value for key, value in row.items() if not key.startswith("_")}
        for index, row in all_indexed
    }
    result_df = pd.DataFrame.from_dict(results_by_idx, orient="index")
    overlap_columns = [column for column in result_df.columns if column in original_df.columns]
    if overlap_columns:
        result_df = result_df.drop(columns=overlap_columns)
    merged = original_df.join(result_df)
    merged_path = os.path.join(run_dir, "merged.csv")
    merged.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"  MERGED CSV : {merged_path}")
    return merged_path


def write_run_info(
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
    usage_totals: UsageDict,
    usage_keys: list[str],
    all_rows: list[ResultRow] | None = None,
    label_field: str | None = None,
    date_range: tuple[str, str] | None = None,
) -> None:
    duration = (end_time - start_time).total_seconds()
    info_path = os.path.join(run_dir, "run_info.txt")
    with open(info_path, "w", encoding="utf-8") as handle:
        handle.write("Run Info\n")
        handle.write("========\n")
        handle.write(f"Date       : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Command    : {_redact_api_key(run_command)}\n")
        handle.write(f"Input      : {os.path.abspath(input_path)}\n")
        handle.write(f"Output dir : {run_dir}\n")
        handle.write(f"Model      : {model}\n")
        handle.write(f"Workers    : {workers}\n")
        handle.write(f"Batch size : {batch_size}\n")
        handle.write(f"Timeout    : {timeout}s\n")
        handle.write(f"Limit      : {limit if limit else 'none (all rows)'}\n")
        handle.write(f"Rows       : {total_rows}\n")
        if date_range:
            handle.write(f"Data range : {date_range[0]} s/d {date_range[1]}\n")
        handle.write(f"Duration   : {duration:.1f}s\n")
        handle.write("\nUsage\n")
        handle.write("=====\n")
        _write_usage_lines(handle, usage_totals, usage_keys)
        if label_field and all_rows:
            handle.write("\nDistribusi Label\n")
            handle.write("================\n")
            _write_label_counts(handle, all_rows, label_field)
    print(f"  RUN INFO   : {info_path}")


def append_retry_info(
    run_dir: str,
    run_command: str,
    total_failed: int,
    start_time: datetime,
    end_time: datetime,
    usage_totals: UsageDict,
    usage_keys: list[str],
    all_rows: list[ResultRow] | None = None,
    label_field: str | None = None,
) -> None:
    duration = (end_time - start_time).total_seconds()
    info_path = os.path.join(run_dir, "run_info.txt")
    with open(info_path, "a", encoding="utf-8") as handle:
        handle.write("\nRetry\n")
        handle.write("=====\n")
        handle.write(f"Date     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Command  : {run_command}\n")
        handle.write(f"Retried  : {total_failed} baris\n")
        handle.write(f"Duration : {duration:.1f}s\n")
        if usage_keys:
            handle.write("Usage\n")
            _write_usage_lines(handle, usage_totals, usage_keys)
        if label_field and all_rows:
            handle.write("\nDistribusi Label\n")
            handle.write("================\n")
            _write_label_counts(handle, all_rows, label_field)
    print(f"  RUN INFO   : {info_path}")


def _write_label_counts(handle: object, all_rows: list[ResultRow], label_field: str) -> None:
    counts: dict[str, int] = {}
    for row in all_rows:
        label = str(row.get(label_field, "") or "").strip() or "(kosong)"
        counts[label] = counts.get(label, 0) + 1
    total = len(all_rows)
    text_handle = handle
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        text_handle.write(f"  {label:<30} {count:>4}  ({pct:.1f}%)\n")
    text_handle.write(f"  {'TOTAL':<30} {total:>4}\n")


def _write_usage_lines(handle: object, usage_totals: UsageDict, usage_keys: list[str]) -> None:
    label_map = {
        "prompt_tokens": "Prompt tokens      ",
        "completion_tokens": "Completion tokens  ",
        "total_tokens": "Total tokens       ",
        "cached_tokens": "Cached tokens      ",
        "cache_write_tokens": "Cache write tokens ",
        "audio_tokens": "Audio tokens       ",
        "reasoning_tokens": "Reasoning tokens   ",
        "cost": "Cost               ",
        "upstream_inference_cost": "Upstream cost      ",
    }
    text_handle = handle
    for key in usage_keys:
        label = label_map[key]
        value = usage_totals.get(key, 0.0)
        if key in {"cost", "upstream_inference_cost"}:
            text_handle.write(f"{label}: {value:.6f}\n")
        else:
            text_handle.write(f"{label}: {int(value)}\n")
