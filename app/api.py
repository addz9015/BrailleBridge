"""
FastAPI backend for Braille recognition deployment.
"""

import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.backend import backend


def _parse_cors_origins() -> list[str]:
    raw_origins = os.getenv(
        "API_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:3000,http://localhost:3000",
    )
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins


app = FastAPI(title="Braille Recognition API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    sample_id: int = Field(ge=0)
    use_lm: bool = True
    lm_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    beam_width: Optional[int] = Field(default=None, ge=1, le=64)


class TapCell(BaseModel):
    dots: list[int] = Field(min_length=6, max_length=6)


class TapAnalyzeRequest(BaseModel):
    word: str = Field(min_length=1, max_length=32)
    cells: list[TapCell] = Field(min_length=1, max_length=32)


def _serialize_result(result: dict) -> dict:
    candidates = [
        {"text": text, "score": float(score)}
        for text, score in result.get("candidates", [])
    ]

    return {
        "sample_id": int(result["sample_id"]),
        "word": result["word"],
        "noisy": np.asarray(result["noisy"], dtype=np.float32).tolist(),
        "denoised": np.asarray(result["denoised"], dtype=np.float32).tolist(),
        "greedy_pred": result["greedy_pred"],
        "lm_pred": result["lm_pred"],
        "candidates": candidates,
        "greedy_ok": bool(result["greedy_ok"]),
        "lm_ok": bool(result["lm_ok"]),
        "lm_weight": float(result["lm_weight"]),
        "beam_width": int(result["beam_width"]),
    }


def _serialize_tap_result(result: dict) -> dict:
    return {
        "word_input": result["word_input"],
        "word_sanitized": result["word_sanitized"],
        "predicted_word": result["predicted_word"],
        "direct_word": result["direct_word"],
        "decode_mode": result["decode_mode"],
        "snapped_similarity": float(result["snapped_similarity"]),
        "confidence": float(result["confidence"]),
        "unknown_cells": int(result["unknown_cells"]),
        "cells": result["cells"],
        "warnings": list(result["warnings"]),
        "correct": bool(result["correct"]),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "samples": len(backend.samples)}


@app.get("/samples")
def sample_options() -> dict:
    return {"choices": backend.sample_options()}


@app.get("/sample/{sample_id}")
def get_sample(sample_id: int) -> dict:
    try:
        return _serialize_result(backend.analyze_sample(sample_id, use_lm=False))
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        result = backend.analyze_sample(
            sample_id=request.sample_id,
            use_lm=request.use_lm,
            lm_weight=request.lm_weight,
            beam_width=request.beam_width,
        )
        return _serialize_result(result)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/tap/analyze")
def tap_analyze(request: TapAnalyzeRequest) -> dict:
    try:
        cells = [cell.dots for cell in request.cells]
        result = backend.analyze_tap_cells(word=request.word, cells=cells)
        return _serialize_tap_result(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
