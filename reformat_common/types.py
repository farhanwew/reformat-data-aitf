from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel

ResultRow = dict[str, Any]
UsageDict = dict[str, float]

GeneratedT = TypeVar("GeneratedT")


@dataclass(frozen=True)
class PreparedRow(Generic[GeneratedT]):
    llm_input: str
    skip_generation: bool = False
    preset_result: GeneratedT | None = None


@dataclass(frozen=True)
class GenerationConfig(Generic[GeneratedT]):
    system_prompt: str
    user_prompt_template: str
    user_prompt_var: str
    required_response_keys: list[str]
    payload_to_result: Callable[[dict[str, str]], GeneratedT]
    primary_mode: Literal["pydantic_parse", "response_format_json"]
    pydantic_model: type[BaseModel] | None = None
    primary_response_format: Any = None
    normalize_payload: Callable[[dict[str, str]], dict[str, str]] | None = None
    fallback_on_non_server_api_error: bool = False
    fallback_response_format: dict[str, str] = field(
        default_factory=lambda: {"type": "json_object"}
    )


@dataclass(frozen=True)
class RunnerConfig(Generic[GeneratedT]):
    output_fields: list[str]
    required_columns: set[str]
    run_dir_suffix: str
    generation: GenerationConfig[GeneratedT]
    prepare_row: Callable[[dict[str, Any]], PreparedRow[GeneratedT]]
    build_result_row: Callable[[dict[str, Any], GeneratedT], ResultRow]
    build_error_result: Callable[[dict[str, Any]], GeneratedT]
    validate_input_df: Callable[[pd.DataFrame], None] | None = None
    empty_run_message: str = "Tidak ada data. Output CSV kosong dibuat."
    usage_report_keys: list[str] = field(default_factory=list)
    retry_usage_report_keys: list[str] = field(default_factory=list)
    label_field: str | None = None
    date_range_field: str | None = None
