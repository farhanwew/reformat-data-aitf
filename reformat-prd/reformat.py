import argparse
import os
import re
import sys
from typing import Any

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from reformat_common import (
    ANALISIS_RESULT_KEYS,
    AnalisisResult,
    GenerationConfig,
    PreparedRow,
    ResultRow,
    RunnerConfig,
    USAGE_KEYS,
    is_na,
    reformat_csv as run_reformat_csv,
    retry_failed_run as run_retry_failed_run,
    single_line,
)

DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_WORKERS = 16
DEFAULT_BATCH_SIZE = 15
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

OUTPUT_FIELDS = ["link", "source", "ringkasan", "klaim", "fakta", "label", "analisis", "date", "path"]

PATH_COLUMN_VARIANTS = ["video path", "video_path", "file name", "filename", "nama file", "nama_file"]

REQUIRED_RESPONSE_KEYS = ANALISIS_RESULT_KEYS


LABEL_MAP = {
    "ujaran kebencian": "UJARAN KEBENCIAN",
    "disinformasi": "DISINFORMASI",
    "fitnah": "FITNAH",
    "netral": "NETRAL",
    "tidak dapat dinilai": "TIDAK DAPAT DINILAI",
}

SYSTEM_PROMPT = """Kamu adalah sistem ekstraksi informasi yang membantu menyusun dataset analisis konten pelanggaran media sosial.

Kamu akan diberikan teks analisis pelanggaran yang ditulis oleh analis manusia. Kadang disertakan juga teks asli konten dari postingan — jika ada, prioritaskan teks asli untuk mengekstrak klaim secara literal.

Tugasmu — ekstrak tiga hal:
1. Ringkasan netral tentang APA isi konten yang dilaporkan.
2. Klaim atau pernyataan spesifik yang ada dalam konten tersebut.
3. Fakta atau bukti penyangkal yang disebutkan (jika ada).
4. Berikan analisis singkat tentang karakteristik kasus.


Aturan ekstraksi — gunakan HANYA informasi yang tersurat dalam teks:
- ringkasan : 1–2 kalimat netral tentang apa yang disampaikan konten (video/postingan).
              Fokus pada ISI konten, bukan kesimpulan pelanggaran analis.
              Jangan gunakan framing penilaian seperti "melanggar", "provokatif", dst.
- klaim     : tulis klaim utama konten dalam 1–2 kalimat factual menggunakan
              sudut pandang ketiga, contoh: "Konten menyatakan bahwa ..." atau
              "Unggahan mengklaim bahwa ...".
              Tulis secara literal — apa yang dikatakan atau ditampilkan — tanpa
              interpretasi, penilaian, atau framing evaluatif.
              Jika teks asli konten tersedia dan relevan, gunakan itu sebagai dasar.
              Jika tidak ada klaim spesifik, tulis "tidak ada detail konten".
- fakta     : tulis FAKTA atau bukti yang menyangkal klaim — fokus pada substansinya:
              apa yang sebenarnya terjadi, data teknis, pernyataan resmi, atau
              klarifikasi dari sumber otoritatif.
              Jangan menyebut nama organisasi fact-checker (Mafindo, TurnBackHoax, dll)
              sebagai sumber utama — tulis faktanya, bukan siapa yang memeriksa.
              Jika tidak ada bukti penyangkal yang konkret, *TULIS* "tidak ada".
- analisis  : jelaskan kenapa konten ini bermasalah — fokus pada apa yang dikatakan/ditampilkan
              dan mengapa hal tersebut berpotensi menimbulkan dampak negatif.
              Jangan sebut nama kategori pelanggarannya secara eksplisit.
              Sertakan kutipan atau konten spesifik sebagai dasar penilaian.
              Gunakan fakta jika ada. Tulis 3–4 kalimat, ringkas dan tidak terlalu panjang.
"""

USER_PROMPT_TEMPLATE = (
    "Ekstrak ringkasan, klaim, dan fakta dari teks berikut.\n\n"
    "{caption}"
)


def map_label(raw_label: str) -> str:
    if not raw_label or is_na(raw_label):
        return "TIDAK DAPAT DINILAI"
    return LABEL_MAP.get(str(raw_label).strip().lower(), str(raw_label).strip().upper())


def clean_analisis(text: str) -> str:
    if not text or is_na(text):
        return ""
    text = re.sub(
        r"Penyebaran informasi.*?(?:Pasal\s[\d\w\s,()dan]+\.?)\s*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def _find_path(row_data: dict[str, Any]) -> str:
    for col in PATH_COLUMN_VARIANTS:
        val = single_line(row_data.get(col, ""))
        if val:
            return val
    return ""


def _build_result_row(
    row_data: dict[str, Any],
    parsed: AnalisisResult,
) -> ResultRow:
    link_value = single_line(row_data.get("url", "")) or single_line(row_data.get("link", ""))
    raw_label = str(row_data.get("kategori", ""))
    raw_analisis = _combine_raw_analisis(row_data)
    return {
        "link": link_value,
        "source": single_line(row_data.get("platform", "")),
        "ringkasan": single_line(parsed.ringkasan),
        "klaim": single_line(parsed.klaim),
        "fakta": single_line(parsed.fakta or "tidak ada"),
        "label": single_line(map_label(raw_label)),
        "analisis": single_line(parsed.analisis or clean_analisis(raw_analisis)),
        "date": single_line(row_data.get("date", "")),
        "path": _find_path(row_data),
    }


def _build_llm_input(row_data: dict[str, Any]) -> str:
    analisis = str(row_data.get("analisis_pelanggaran", "")).strip()
    dampak = str(row_data.get("analisis_dampak", "")).strip()
    caption_post = str(row_data.get("CAPTION_POST", "")).strip()
    caption_status = str(row_data.get("CAPTION_STATUS", "")).strip().lower()
    raw_label = str(row_data.get("kategori", "")).strip()
    label = map_label(raw_label) if raw_label and raw_label.lower() != "nan" else ""

    label_section = f"Label pelanggaran: {label}\n\n" if label else ""
    dampak_section = (
        f'\n\nAnalisis dampak:\n"""\n{dampak}\n"""'
        if dampak and dampak.lower() != "nan"
        else ""
    )
    caption_section = (
        f'\n\nTeks asli konten:\n"""\n{caption_post}\n"""'
        if caption_status == "ok" and caption_post and caption_post.lower() != "nan"
        else ""
    )
    return f'{label_section}Analisis pelanggaran:\n"""\n{analisis}\n"""{dampak_section}{caption_section}'


def _prepare_row(row_data: dict[str, Any]) -> PreparedRow[AnalisisResult]:
    analisis_raw = str(row_data.get("analisis_pelanggaran", "")).strip()
    if not analisis_raw or analisis_raw.lower() == "nan":
        return PreparedRow(
            llm_input="",
            skip_generation=True,
            preset_result=AnalisisResult(
                ringkasan="caption tidak tersedia",
                klaim="tidak ada detail konten",
                fakta="tidak ada",
                analisis="",
            ),
        )
    return PreparedRow(llm_input=_build_llm_input(row_data))


def _build_error_result(_: dict[str, Any]) -> AnalisisResult:
    return AnalisisResult(
        ringkasan="tidak dapat diproses",
        klaim="tidak dapat diproses",
        fakta="tidak ada",
        analisis="tidak dapat diproses",
    )


def _combine_raw_analisis(row_data: dict[str, Any]) -> str:
    parts = []
    for column in ("analisis_pelanggaran", "analisis_dampak"):
        value = str(row_data.get(column, "")).strip()
        if value and value.lower() != "nan":
            parts.append(value)
    return " ".join(parts)


def _validate_input_df(df: pd.DataFrame) -> None:
    if "url" not in df.columns and "link" not in df.columns:
        raise ValueError("CSV input wajib punya salah satu kolom: url atau link")


CONFIG = RunnerConfig(
    output_fields=OUTPUT_FIELDS,
    required_columns={"platform", "analisis_pelanggaran", "kategori"},
    run_dir_suffix="prd",
    validate_input_df=_validate_input_df,
    generation=GenerationConfig(
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        user_prompt_var="caption",
        required_response_keys=REQUIRED_RESPONSE_KEYS,
        payload_to_result=AnalisisResult.model_validate,
        primary_mode="pydantic_parse",
        pydantic_model=AnalisisResult,
        fallback_on_non_server_api_error=True,
    ),
    prepare_row=_prepare_row,
    build_result_row=_build_result_row,
    build_error_result=_build_error_result,
    usage_report_keys=USAGE_KEYS,
    retry_usage_report_keys=USAGE_KEYS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat CSV ke format ringkasan/klaim/fakta/label/analisis"
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
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument("--model", help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Jumlah thread paralel (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Baris per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout dalam detik (default: 60)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Proses hanya N baris pertama (untuk testing)",
    )
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
