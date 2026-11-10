"""Customer Churn Prediction System API and single-page frontend.

Run from project root: uvicorn app.main:app --reload
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.database import fetch_history, initialize, record
from app.schemas import CustomerInput, HistoryResponse, PredictionResponse
from ml.common import API_TO_DATASET, FEATURES, MODEL_FILE, MODEL_METADATA, ROOT

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format=LOG_FORMAT)
logger = logging.getLogger("churn_api")
FRONTEND_DIR = Path(__file__).resolve().parent / "static"


def risk_level(probability: float, low_below: float = 0.30, high_at_or_above: float = 0.70) -> str:
    if probability < low_below:
        return "Low"
    if probability >= high_at_or_above:
        return "High"
    return "Medium"


def create_app(*, model_path: Path | None = None, metadata_path: Path | None = None,
               database_path: Path | None = None) -> FastAPI:
    model_path = Path(model_path or os.getenv("MODEL_PATH", MODEL_FILE))
    metadata_path = Path(metadata_path or os.getenv("MODEL_METADATA_PATH", MODEL_METADATA))
    database_path = Path(database_path or os.getenv("PREDICTION_DB_PATH", ROOT / "runtime" / "history.sqlite3"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = None
        app.state.metadata = None
        if model_path.is_file() and metadata_path.is_file():
            try:
                # Load ONLY trusted local model files; joblib uses pickle serialization.
                app.state.pipeline = joblib.load(model_path)
                app.state.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                logger.info("Model loaded: %s", app.state.metadata.get("best_model"))
            except Exception:
                logger.exception("Failed to load model files")
        else:
            logger.error("Trained model missing; execute 'python -m ml.train'")
        initialize(database_path)
        logger.info("History database ready: %s", database_path)
        yield

    app = FastAPI(
        title="Customer Churn Prediction System API",
        description="Scikit-learn churn prediction with probabilities, risk categories, and SQLite history.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    reports_figures = ROOT / "reports" / "figures"
    reports_figures.mkdir(parents=True, exist_ok=True)
    app.mount("/figures", StaticFiles(directory=reports_figures), name="figures")

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                   for e in exc.errors()]
        logger.warning("validation_error path=%s fields=%s", request.url.path,
                       [d["field"] for d in details])
        return JSONResponse(status_code=422, content={
            "error": {"code": "VALIDATION_ERROR", "message": "Invalid input.", "details": details}
        })

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        logger.warning("http_error status=%s path=%s", exc.status_code, request.url.path)
        return JSONResponse(status_code=exc.status_code, content={
            "error": {"code": "HTTP_ERROR", "message": str(exc.detail)}
        })

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        # Never expose raw customer inputs or internal stack traces in responses.
        logger.exception("internal_error path=%s", request.url.path)
        return JSONResponse(status_code=500, content={
            "error": {"code": "INTERNAL_ERROR", "message": "An unexpected server error occurred."}
        })

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/health", tags=["system"])
    def health(request: Request):
        loaded = request.app.state.pipeline is not None
        return JSONResponse(
            status_code=200 if loaded else 503,
            content={"status": "ok" if loaded else "unavailable", "model_loaded": loaded},
        )

    @app.get("/model/info", tags=["model"])
    def model_info(request: Request):
        info = request.app.state.metadata
        if info is None:
            raise HTTPException(status_code=503, detail="Model unavailable. Run training first.")
        return {
            "model_name": info["best_model"],
            "trained_at_utc": info["trained_at_utc"],
            "features": info["feature_names"],
            "target": "Churn",
            "decision_threshold": info["decision_threshold"],
            "risk_thresholds": info["risk_thresholds"],
            "test_metrics": info["test_metrics"],
            "validation_results": info["validation_results"],
            "train_rows": info["training"]["usable_rows"],
            "test_rows": info["testing"]["usable_rows"],
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
    def predict(payload: CustomerInput, request: Request):
        model = request.app.state.pipeline
        info = request.app.state.metadata
        if model is None or info is None:
            raise HTTPException(status_code=503, detail="Model unavailable. Run training first.")
        # Pydantic strictly validates the API input; this maps keys to training columns.
        row = {API_TO_DATASET[field]: value for field, value in payload.model_dump().items()}
        frame = pd.DataFrame([row], columns=FEATURES)
        churn_index = list(model.classes_).index(1)
        probability = float(model.predict_proba(frame)[0, churn_index])
        if not np.isfinite(probability) or probability < 0 or probability > 1:
            raise RuntimeError("Model returned an invalid probability")
        outcome = "Churn" if probability >= info["decision_threshold"] else "No Churn"
        limits = info["risk_thresholds"]
        level = risk_level(probability, limits["low_below"], limits["high_at_or_above"])
        result = PredictionResponse(prediction=outcome, probability=round(probability, 6), risk_level=level)
        record(database_path, outcome, probability, level)
        logger.info("prediction_completed outcome=%s risk=%s", outcome, level)
        return result

    @app.get("/predictions", response_model=HistoryResponse, tags=["predictions"])
    def predictions(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
        return fetch_history(database_path, limit, offset)

    return app


app = create_app()
