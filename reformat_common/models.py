from __future__ import annotations

from pydantic import BaseModel

ANALISIS_RESULT_KEYS = ["ringkasan", "klaim", "fakta", "analisis"]
FACTCHECK_RESULT_KEYS = ["ringkasan", "klaim", "fakta", "label", "analisis"]


class AnalisisResult(BaseModel):
    ringkasan: str
    klaim: str
    fakta: str
    analisis: str


class FactcheckResult(AnalisisResult):
    label: str
