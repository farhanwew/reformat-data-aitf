import argparse
import os
import sys
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from reformat_common import (
    FACTCHECK_RESULT_KEYS,
    FactcheckResult,
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

# DEFAULT_MODEL = "github_copilot/gpt-5-mini"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_WORKERS = 16
DEFAULT_BATCH_SIZE = 15
# DEFAULT_BASE_URL = "http://localhost:4000"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"



DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

OUTPUT_FIELDS = ["link", "video_name", "source", "ringkasan", "klaim", "fakta", "label", "analisis"]
VALID_LABELS = {"UJARAN KEBENCIAN", "DISINFORMASI", "FITNAH", "NETRAL", "TIDAK DAPAT DINILAI"}
REQUIRED_RESPONSE_KEYS = FACTCHECK_RESULT_KEYS

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
- analisis  : Jelaskan karakteristik kasus berdasarkan klaim dan fakta yang ditemukan. fokus pada apa yang diklaim, apakah ada
              bukti yang mendukung atau menyangkal, serta dampak potensial yang dapat ditimbulkan.
              Jangan sebut nama kategori pelanggarannya secara eksplisit dan Jangan sebut nama label di dalam teks analisis.
              Sertakan kutipan atau konten spesifik sebagai dasar penilaian, dan gunakan fakta jika tersedia.
              Tulis dalam 2–3 kalimat secara ringkas, tidak terlalu panjang maupun terlalu singkat.
"""

USER_PROMPT_TEMPLATE = (
    "Analisis konten berikut dan tentukan ringkasan, klaim, fakta, label, dan analisis.\n\n"
    "{content}\n\n"
    "PENTING: Kembalikan output dalam JSON object valid saja dengan tepat 5 key berikut: "
    '"ringkasan", "klaim", "fakta", "label", "analisis". '
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
    parts = []
    narasi = single_line(row_data.get("narasi", ""))
    penjelasan = single_line(row_data.get("penjelasan", ""))
    kesimpulan = single_line(row_data.get("kesimpulan", ""))

    if narasi:
        parts.append(f'Narasi:\n"""\n{narasi}\n"""')
    if penjelasan:
        parts.append(f'Penjelasan:\n"""\n{penjelasan}\n"""')
    if kesimpulan:
        parts.append(f'Kesimpulan:\n"""\n{kesimpulan}\n"""')
    return "\n\n".join(parts)


def _build_result_row(
    row_data: dict[str, Any],
    generated: FactcheckResult,
) -> ResultRow:
    link = single_line(row_data.get("link_video_asli", "")) or single_line(row_data.get("link_article", ""))
    source = infer_platform(link) if link else ""
    return {
        "link": link,
        "video_name": single_line(row_data.get("video_name", "")),
        "source": source,
        "ringkasan": single_line(generated.ringkasan),
        "klaim": single_line(generated.klaim),
        "fakta": single_line(generated.fakta) or "tidak ada",
        "label": _normalize_label(generated.label),
        "analisis": single_line(generated.analisis),
    }


def _prepare_row(row_data: dict[str, Any]) -> PreparedRow[FactcheckResult]:
    content = _build_llm_input(row_data)
    if not content.strip():
        return PreparedRow(
            llm_input="",
            skip_generation=True,
            preset_result=FactcheckResult(
                ringkasan="konten tidak tersedia",
                klaim="tidak ada detail konten",
                fakta="tidak ada",
                label="TIDAK DAPAT DINILAI",
                analisis="",
            ),
        )
    return PreparedRow(llm_input=content)


def _build_error_result(_: dict[str, Any]) -> FactcheckResult:
    return FactcheckResult(
        ringkasan="tidak dapat diproses",
        klaim="tidak dapat diproses",
        fakta="tidak ada",
        label="TIDAK DAPAT DINILAI",
        analisis="tidak dapat diproses",
    )


CONFIG = RunnerConfig(
    output_fields=OUTPUT_FIELDS,
    required_columns={"narasi", "link_video_asli"},
    run_dir_suffix="tbh",
    generation=GenerationConfig(
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        user_prompt_var="content",
        required_response_keys=REQUIRED_RESPONSE_KEYS,
        payload_to_result=FactcheckResult.model_validate,
        primary_mode="pydantic_parse",
        pydantic_model=FactcheckResult,
        normalize_payload=_normalize_payload,
        fallback_on_non_server_api_error=True,
    ),
    prepare_row=_prepare_row,
    build_result_row=_build_result_row,
    build_error_result=_build_error_result,
    label_field="label",
    date_range_field="date",
    usage_report_keys=USAGE_KEYS,
    retry_usage_report_keys=USAGE_KEYS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat fact-check CSV ke format ringkasan/klaim/fakta/label/analisis"
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
