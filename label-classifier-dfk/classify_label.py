import argparse
import os
import sys
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pydantic import BaseModel

from reformat_common import (
    GenerationConfig,
    PreparedRow,
    ResultRow,
    RunnerConfig,
    USAGE_KEYS,
    infer_platform,
    reformat_csv as run_reformat_csv,
    retry_failed_run as run_retry_failed_run,
    single_line,
)

DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_WORKERS = 16
DEFAULT_BATCH_SIZE = 15
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

OUTPUT_FIELDS = ["link", "label_asal", "label", "alasan"]
VALID_LABELS = {"UJARAN KEBENCIAN", "DISINFORMASI", "FITNAH"}
REQUIRED_RESPONSE_KEYS = ["label", "alasan"]


class LabelResult(BaseModel):
    label: str
    alasan: str


SYSTEM_PROMPT = """Kamu adalah sistem klasifikasi konten media sosial berdasarkan regulasi ITE Indonesia.

Kamu akan diberikan ringkasan atau konten dari sebuah postingan media sosial.

Tugasmu — tentukan SATU label kategori yang paling sesuai dan berikan alasan singkat.

Label yang tersedia (pilih SATU):
- UJARAN KEBENCIAN :
  * (UU No. 1 Tahun 2024 - Pasal 28 ayat 2)
  * Menyebarkan informasi yang bertujuan menimbulkan rasa kebencian atau permusuhan
    individu/kelompok berdasarkan SARA.

- FITNAH :
  * (UU No. 1 Tahun 2024 - Pasal 27A)
  * Menyerang kehormatan/nama baik dengan menuduhkan hal tertentu melalui sistem
    elektronik agar diketahui umum.

- DISINFORMASI :
  * (UU No. 1 Tahun 2024 - Pasal 28 ayat 1 & 3)
  * Menyebarkan berita bohong/menyesatkan yang mengakibatkan kerugian atau kerusuhan.
  * Termasuk konten yang secara eksplisit menghasut, memprovokasi, atau mendorong
    tindakan kekerasan, kerusuhan, penjarahan, atau perusakan fasilitas publik —
    meskipun tidak mengandung informasi yang secara teknis "bohong".


Aturan:
- label  : tulis PERSIS salah satu dari tiga label di atas (huruf kapital semua).
- alasan : jelaskan dalam 2–4 kalimat mengapa label tersebut dipilih berdasarkan konten.
           Jangan sebut nama label secara eksplisit dalam kalimat alasan.
"""

USER_PROMPT_TEMPLATE = (
    "Klasifikasikan konten berikut.\n\n"
    "{content}\n\n"
    "PENTING: Kembalikan output dalam JSON object valid saja dengan tepat 2 key berikut: "
    '"label", "alasan". '
    "Jangan tambahkan teks lain di luar JSON."
)


def _normalize_label(raw: str) -> str:
    cleaned = raw.strip().upper()
    return cleaned if cleaned in VALID_LABELS else "TIDAK DAPAT DINILAI"


def _normalize_payload(payload: dict[str, str]) -> dict[str, str]:
    normalized = dict(payload)
    normalized["label"] = _normalize_label(normalized.get("label", ""))
    return normalized


def _build_llm_input(row_data: dict[str, Any]) -> str:
    raw_label = single_line(row_data.get("label", ""))
    label_header = f"Label saat ini: {raw_label}\n\n" if raw_label else ""

    parts = []

    # Priority 1: reformat-prd output fields
    ringkasan = single_line(row_data.get("ringkasan", ""))
    klaim = single_line(row_data.get("klaim", ""))
    fakta = single_line(row_data.get("fakta", ""))
    analisis = single_line(row_data.get("analisis", ""))
    if any([ringkasan, klaim, fakta, analisis]):
        if ringkasan:
            parts.append(f'Ringkasan:\n"""\n{ringkasan}\n"""')
        if klaim:
            parts.append(f'Klaim:\n"""\n{klaim}\n"""')
        if fakta and fakta.lower() != "tidak ada":
            parts.append(f'Fakta:\n"""\n{fakta}\n"""')
        if analisis:
            parts.append(f'Analisis:\n"""\n{analisis}\n"""')
        return label_header + "\n\n".join(parts)

    # Priority 2: TBH / fact-check fields
    narasi = single_line(row_data.get("narasi", ""))
    penjelasan = single_line(row_data.get("penjelasan", ""))
    kesimpulan = single_line(row_data.get("kesimpulan", ""))
    if any([narasi, penjelasan, kesimpulan]):
        if narasi:
            parts.append(f'Narasi:\n"""\n{narasi}\n"""')
        if penjelasan:
            parts.append(f'Penjelasan:\n"""\n{penjelasan}\n"""')
        if kesimpulan:
            parts.append(f'Kesimpulan:\n"""\n{kesimpulan}\n"""')
        return label_header + "\n\n".join(parts)

    # Priority 3: raw prd fields
    analisis_prd = single_line(row_data.get("analisis_pelanggaran", ""))
    if analisis_prd:
        return f'{label_header}Analisis pelanggaran:\n"""\n{analisis_prd}\n"""'

    return ""


def _build_result_row(row_data: dict[str, Any], generated: LabelResult) -> ResultRow:
    link = single_line(row_data.get("link", "")) or single_line(row_data.get("url", ""))
    return {
        "link": link,
        "label_asal": single_line(row_data.get("label", "")),
        "label": _normalize_label(generated.label),
        "alasan": single_line(generated.alasan),
    }


def _prepare_row(row_data: dict[str, Any]) -> PreparedRow[LabelResult]:
    raw_label = single_line(row_data.get("label", "")).upper()

    if raw_label in VALID_LABELS:
        return PreparedRow(
            llm_input="",
            skip_generation=True,
            preset_result=LabelResult(label=raw_label, alasan=""),
        )

    llm_input = _build_llm_input(row_data)
    if not llm_input:
        return PreparedRow(
            llm_input="",
            skip_generation=True,
            preset_result=LabelResult(label="TIDAK DAPAT DINILAI", alasan=""),
        )

    return PreparedRow(llm_input=llm_input)


def _build_error_result(_: dict[str, Any]) -> LabelResult:
    return LabelResult(label="TIDAK DAPAT DINILAI", alasan="tidak dapat diproses")


CONFIG = RunnerConfig(
    output_fields=OUTPUT_FIELDS,
    required_columns={"label"},
    run_dir_suffix="classify",
    generation=GenerationConfig(
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        user_prompt_var="content",
        required_response_keys=REQUIRED_RESPONSE_KEYS,
        payload_to_result=LabelResult.model_validate,
        primary_mode="pydantic_parse",
        pydantic_model=LabelResult,
        normalize_payload=_normalize_payload,
        fallback_on_non_server_api_error=True,
    ),
    prepare_row=_prepare_row,
    build_result_row=_build_result_row,
    build_error_result=_build_error_result,
    label_field="label",
    merge_output_fields=["label"],
    usage_report_keys=USAGE_KEYS,
    retry_usage_report_keys=USAGE_KEYS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Klasifikasi label konten — skip jika sudah valid, examine jika tidak dikenal"
    )
    parser.add_argument("input", nargs="?", help="Path file CSV input")
    parser.add_argument(
        "output",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder output (default: {DEFAULT_OUTPUT_DIR})",
    )
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
        run_retry_failed_run(
            CONFIG,
            run_dir=args.retry_failed,
            api_key=key,
            base_url=args.base_url,
            model=model,
            workers=args.workers,
            batch_size=args.batch_size,
            timeout=args.timeout,
            run_command=" ".join(sys.argv),
        )
        return

    if not args.input:
        parser.error("input wajib diisi (atau gunakan --retry-failed)")

    run_reformat_csv(
        CONFIG,
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
