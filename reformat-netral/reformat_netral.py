import argparse
import os
import sys
from typing import Any

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
    infer_platform,
    reformat_csv as run_reformat_csv,
    retry_failed_run as run_retry_failed_run,
    single_line,
)

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 15
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

OUTPUT_FIELDS = ["link", "source", "ringkasan", "klaim", "fakta", "label", "analisis"]
REQUIRED_RESPONSE_KEYS = ANALISIS_RESULT_KEYS


SYSTEM_PROMPT = """Kamu adalah sistem analisis konten media sosial yang membantu menyusun dataset.

Kamu akan diberikan caption/teks dari sebuah konten dari media sosoial.

Tugasmu — ekstrak empat hal:
1. Ringkasan netral tentang APA isi konten.
2. Klaim atau pernyataan utama yang disampaikan konten.
3. Fakta atau informasi faktual yang ada di dalam teks.
4. Analisis singkat mengenai isi konten yang dimuat.

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

- analisis  : Analisis isi  post dalam 3–4 kalimat padat dengan struktur:
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


def _build_result_row(row_data: dict[str, Any], parsed: AnalisisResult) -> ResultRow:
    url = single_line(row_data.get("url", ""))
    return {
        "link": url,
        "source": infer_platform(url) if url else single_line(row_data.get("creator", "")),
        "ringkasan": single_line(parsed.ringkasan),
        "klaim": single_line(parsed.klaim),
        "fakta": single_line(parsed.fakta) or "tidak ada",
        "label": "NETRAL",
        "analisis": single_line(parsed.analisis),
    }


def _prepare_row(row_data: dict[str, Any]) -> PreparedRow[AnalisisResult]:
    caption = single_line(row_data.get("caption", ""))
    if not caption:
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
    return PreparedRow(llm_input=caption)


def _build_error_result(_: dict[str, Any]) -> AnalisisResult:
    return AnalisisResult(
        ringkasan="tidak dapat diproses",
        klaim="tidak dapat diproses",
        fakta="tidak ada",
        analisis="tidak dapat diproses",
    )


CONFIG = RunnerConfig(
    output_fields=OUTPUT_FIELDS,
    required_columns={"caption", "url"},
    run_dir_suffix="netral",
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
    empty_run_message="Tidak ada data.",
    usage_report_keys=[
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "upstream_inference_cost",
    ],
    retry_usage_report_keys=["cost"],
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat netral CSV ke format ringkasan/klaim/fakta/label/analisis"
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
