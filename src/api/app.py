"""FastAPI app for live deception-risk inference."""

from __future__ import annotations

from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.api.live_logger import LiveInferenceLogger
from src.api.live_session import LiveInferenceSession
from src.api.schemas import GazeSample, PredictionRequest, PredictionResponse, StreamInputMessage
from src.realtime.decision_policy import risk_category
from src.realtime.model_registry import check_required_files
from src.realtime.model_selection_config import (
    DISABLED_MODELS,
    EXPERIMENTAL_MODELS,
    FALLBACK_MODEL,
    SELECTED_MODEL,
)
from src.realtime.realtime_config import (
    MIN_VALID_RATIO,
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    UPDATE_INTERVAL_SECONDS,
    WINDOW_SIZE_SECONDS,
)
from src.realtime.realtime_predictor import RealtimePredictor


app = FastAPI(title="Eye-Tracking Deception Risk Live Inference API")
app.mount("/static", StaticFiles(directory="web"), name="static")

logger = LiveInferenceLogger()
live_sessions: dict[str, LiveInferenceSession] = {}
_predictor: RealtimePredictor | None = None
_model_available = False
_model_error = ""


@app.on_event("startup")
def startup_event() -> None:
    """Validate selected model files and write startup report."""
    global _model_available, _model_error
    try:
        status = check_required_files(SELECTED_MODEL)
        _model_available = bool(status["all_files_exist"])
        _model_error = "" if _model_available else "Missing files: " + ", ".join(status["missing_files"])
    except Exception as exc:
        _model_available = False
        _model_error = str(exc)
    logger.write_api_run_report(
        {
            "selected_model": SELECTED_MODEL,
            "model_available": _model_available,
            "model_error": _model_error or "none",
            "note": "This service estimates deception risk under a controlled experimental protocol.",
        }
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "Eye-Tracking Deception Risk Live Inference API",
        "status": "running",
        "selected_model": SELECTED_MODEL,
        "note": "This service estimates deception risk under a controlled experimental protocol. It is not a universal lie detector.",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    return {
        "selected_model": SELECTED_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "model_available": _model_available,
        "disabled_models": DISABLED_MODELS,
        "experimental_models": EXPERIMENTAL_MODELS,
        "risk_thresholds": {
            "low": RISK_LOW_THRESHOLD,
            "high": RISK_HIGH_THRESHOLD,
        },
        "note": "Prototype thresholds are preliminary.",
        "model_error": _model_error or None,
    }


@app.post("/predict/window", response_model=PredictionResponse)
def predict_window(request: PredictionRequest):
    """Predict risk for one provided current gaze window."""
    if not _model_available:
        return PredictionResponse(
            model_type=SELECTED_MODEL,
            probability=0.0,
            smoothed_probability=None,
            risk_category="insufficient_data",
            valid_ratio=0.0,
            sample_count=0,
            latency_ms=0.0,
            status="error",
            message=f"Selected model is unavailable: {_model_error}",
        )
    if not request.samples:
        return PredictionResponse(
            model_type=SELECTED_MODEL,
            probability=0.0,
            smoothed_probability=None,
            risk_category="insufficient_data",
            valid_ratio=0.0,
            sample_count=0,
            latency_ms=0.0,
            status="error",
            message="No samples provided.",
        )
    try:
        predictor = _get_predictor()
        rows = []
        for sample in request.samples:
            row = sample.model_dump()
            row["trial_id"] = request.trial_id or row.get("trial_id") or "rest_window"
            rows.append(row)
        window_df = pd.DataFrame(rows)
        current_timestamp = float(window_df["timestamp"].max())
        prediction = predictor.predict(window_df, current_timestamp=current_timestamp)
        category = risk_category(prediction["probability"], prediction["valid_ratio"], MIN_VALID_RATIO)
        response = PredictionResponse(
            model_type=SELECTED_MODEL,
            probability=prediction["probability"],
            smoothed_probability=prediction["probability"],
            risk_category=category,
            valid_ratio=prediction["valid_ratio"],
            sample_count=prediction["sample_count"],
            latency_ms=prediction["latency_ms"],
            status="ok",
            message="Prediction generated.",
        )
        logger.log_prediction(
            {
                "session_id": request.session_id or "",
                "trial_id": request.trial_id or "",
                "model_type": SELECTED_MODEL,
                "timestamp": current_timestamp,
                "probability": response.probability,
                "smoothed_probability": response.smoothed_probability,
                "risk_category": response.risk_category,
                "valid_ratio": response.valid_ratio,
                "sample_count": response.sample_count,
                "latency_ms": response.latency_ms,
                "status": response.status,
            }
        )
        return response
    except Exception as exc:
        return PredictionResponse(
            model_type=SELECTED_MODEL,
            probability=0.0,
            smoothed_probability=None,
            risk_category="insufficient_data",
            valid_ratio=0.0,
            sample_count=0,
            latency_ms=0.0,
            status="error",
            message=f"Prediction failed: {exc}",
        )


@app.post("/session/reset")
def reset_sessions() -> dict:
    live_sessions.clear()
    return {"status": "ok", "message": "Sessions reset."}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = f"live_{uuid4().hex[:8]}"
    session: LiveInferenceSession | None = None
    try:
        if not _model_available:
            await websocket.send_json({"type": "error", "message": f"Selected model is unavailable: {_model_error}"})
        while True:
            raw_message = await websocket.receive_json()
            try:
                message = StreamInputMessage.model_validate(raw_message)
            except ValidationError as exc:
                await websocket.send_json({"type": "error", "message": f"Invalid message: {exc.errors()}"})
                continue

            if message.type == "start_session":
                session_id = message.session_id or session_id
                session = _create_live_session(session_id)
                live_sessions[session_id] = session
                await websocket.send_json({"type": "session_started", "session_id": session_id, "status": "ok"})
            elif message.type == "start_trial":
                session = session or _create_live_session(session_id)
                live_sessions[session_id] = session
                trial_id = message.trial_id or "live_trial"
                session.start_trial(trial_id)
                await websocket.send_json({"type": "trial_started", "session_id": session_id, "trial_id": trial_id, "status": "ok"})
            elif message.type == "sample":
                if not _model_available:
                    await websocket.send_json({"type": "error", "message": f"Selected model is unavailable: {_model_error}"})
                    continue
                session = session or _create_live_session(session_id)
                live_sessions[session_id] = session
                if message.data is None:
                    await websocket.send_json({"type": "error", "message": "Sample message missing data."})
                    continue
                try:
                    sample = GazeSample.model_validate(message.data).model_dump()
                except ValidationError as exc:
                    await websocket.send_json({"type": "error", "message": f"Invalid sample: {exc.errors()}"})
                    continue
                prediction = session.add_sample(sample)
                if prediction is not None:
                    logger.log_prediction(prediction)
                    await websocket.send_json(prediction)
            elif message.type == "end_trial":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No active session."})
                    continue
                await websocket.send_json(session.end_trial())
            elif message.type == "end_session":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No active session."})
                    continue
                summary = session.end_session()
                logger.log_session_summary(summary)
                await websocket.send_json(summary)
            elif message.type == "reset":
                if session is not None:
                    session.reset()
                await websocket.send_json({"type": "reset", "session_id": session_id, "status": "ok"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"WebSocket error: {exc}"})


def _get_predictor() -> RealtimePredictor:
    global _predictor
    if _predictor is None:
        _predictor = RealtimePredictor(SELECTED_MODEL)
        _predictor.load_model()
    return _predictor


def _create_live_session(session_id: str) -> LiveInferenceSession:
    return LiveInferenceSession(
        session_id=session_id,
        model_type=SELECTED_MODEL,
        update_interval_seconds=UPDATE_INTERVAL_SECONDS,
        window_size_seconds=WINDOW_SIZE_SECONDS,
    )
