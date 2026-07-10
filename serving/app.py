"""FastAPI demo server: upload a car photo, get predicted year + price plus
a Grad-CAM overlay.

Endpoints
---------
GET  /              -> serves static/index.html (minimal upload page)
GET  /health        -> {"status": "ok", "model_loaded": bool}
POST /predict        -> multipart image upload -> JSON prediction (+ base64 Grad-CAM PNG)

Model loading is graceful: if no exported model is found on disk, the
server still starts (useful for early smoke-testing of the API surface /
frontend before phase 3 training + phase 4 ONNX export are done); /predict
then returns a 503 with a clear message instead of crashing.

Model backends:
  - ONNX (preferred for CPU deploy): fast inference via onnxruntime, but
    this app does NOT compute Grad-CAM from the ONNX graph (autograd isn't
    available on an ONNX Runtime session). See TODO(phase 4) below.
  - Torch (CPU): slower but supports Grad-CAM via
    car_price_vision.interpret.gradcam.GradCAM.

If both are present, ONNX is used for the point prediction and, if a torch
checkpoint is *also* available, it is used only to additionally compute the
Grad-CAM overlay. If only one backend is available, that one does both
(or Grad-CAM is simply omitted for ONNX-only deployments).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger("car_price_vision.serving")
logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).parent / "static"

# TODO(phase 4): point these at the real exported artifacts. Overridable via
# env vars so the same image works for local smoke tests and the VPS deploy.
ONNX_MODEL_PATH = Path(os.environ.get("CPV_ONNX_MODEL_PATH", "model.onnx"))
TORCH_CHECKPOINT_PATH = Path(os.environ.get("CPV_TORCH_CHECKPOINT_PATH", "model.pt"))
BACKBONE_NAME = os.environ.get("CPV_BACKBONE", "convnext_tiny")
IMG_SIZE = int(os.environ.get("CPV_IMG_SIZE", "224"))


class PredictionResponse(BaseModel):
    year: float
    year_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    price_gbp: float
    price_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    gradcam_png_base64: Optional[str] = None
    backend: str
    note: Optional[str] = None


class ModelBundle:
    """Lazily loads and holds whichever model backend(s) are available."""

    def __init__(self) -> None:
        self.onnx_session = None
        self.torch_model = None
        self.gradcam_target_layer = None
        self._load()

    def _load(self) -> None:
        # -- ONNX backend (preferred for CPU inference latency) ------------
        if ONNX_MODEL_PATH.exists():
            try:
                import onnxruntime as ort

                self.onnx_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
                logger.info("Loaded ONNX model from %s", ONNX_MODEL_PATH)
            except Exception:
                logger.exception("Found %s but failed to load it as an ONNX model.", ONNX_MODEL_PATH)
        else:
            logger.warning(
                "No ONNX model found at %s (set CPV_ONNX_MODEL_PATH). "
                "TODO(phase 4): export a trained checkpoint to ONNX.",
                ONNX_MODEL_PATH,
            )

        # -- Torch backend (needed for Grad-CAM; optional fallback for predict) --
        if TORCH_CHECKPOINT_PATH.exists():
            try:
                import torch

                from car_price_vision.models.multitask import MultiTaskCarNet

                model = MultiTaskCarNet(backbone_name=BACKBONE_NAME, pretrained=False)
                checkpoint = torch.load(TORCH_CHECKPOINT_PATH, map_location="cpu")
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self.torch_model = model
                self.gradcam_target_layer = model.backbone.default_target_layer()
                logger.info("Loaded torch checkpoint from %s", TORCH_CHECKPOINT_PATH)
            except Exception:
                logger.exception("Found %s but failed to load it as a torch checkpoint.", TORCH_CHECKPOINT_PATH)
        else:
            logger.warning(
                "No torch checkpoint found at %s (set CPV_TORCH_CHECKPOINT_PATH). "
                "TODO(phase 3): finish training and save a checkpoint here for Grad-CAM support.",
                TORCH_CHECKPOINT_PATH,
            )

    @property
    def is_ready(self) -> bool:
        return self.onnx_session is not None or self.torch_model is not None

    @property
    def can_gradcam(self) -> bool:
        return self.torch_model is not None


app = FastAPI(title="car-price-vision demo", description="Predict a car's year & price from a photo.")
model_bundle: ModelBundle | None = None


@app.on_event("startup")
def _load_model() -> None:
    global model_bundle
    model_bundle = ModelBundle()


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found.")
    return FileResponse(index_path)


@app.get("/health")
def health() -> JSONResponse:
    ready = model_bundle.is_ready if model_bundle is not None else False
    return JSONResponse({"status": "ok", "model_loaded": ready})


def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize/normalize a PIL image into a (1, 3, H, W) float32 array using
    the same ImageNet stats as data/transforms.py:eval_transforms.
    """
    from car_price_vision.data.transforms import eval_transforms

    tensor = eval_transforms(IMG_SIZE)(image)
    return tensor.unsqueeze(0).numpy()


def _predict_onnx(bundle: ModelBundle, image: Image.Image) -> dict:
    input_array = _preprocess(image)
    input_name = bundle.onnx_session.get_inputs()[0].name
    outputs = bundle.onnx_session.run(None, {input_name: input_array.astype(np.float32)})
    # TODO(phase 4): confirm output order once the real ONNX export exists;
    # assumed here to be [year, log_price] matching MultiTaskCarNet.forward().
    year_pred, log_price_pred = float(outputs[0].squeeze()), float(outputs[1].squeeze())
    return {"year": year_pred, "price_gbp": float(np.exp(log_price_pred))}


def _predict_and_gradcam_torch(bundle: ModelBundle, image: Image.Image) -> tuple[dict, Optional[str]]:
    import torch

    from car_price_vision.interpret.gradcam import GradCAM, overlay_heatmap

    tensor = torch.from_numpy(_preprocess(image))

    with torch.no_grad():
        preds = bundle.torch_model(tensor)
    result = {"year": float(preds["year"].item()), "price_gbp": float(np.exp(preds["log_price"].item()))}

    gradcam_b64 = None
    try:
        cam_extractor = GradCAM(bundle.torch_model, bundle.gradcam_target_layer)
        cam = cam_extractor(tensor, output_key="log_price")[0]
        cam_extractor.remove_hooks()
        overlay = overlay_heatmap(image, cam)
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.exception("Grad-CAM computation failed; returning prediction without it.")

    return result, gradcam_b64


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if model_bundle is None or not model_bundle.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model is loaded on this server yet. "
                "TODO(phase 3/4): train a model and export it to "
                f"{ONNX_MODEL_PATH} and/or {TORCH_CHECKPOINT_PATH}."
            ),
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file as an image: {exc}") from exc

    gradcam_b64 = None
    note = None

    if model_bundle.torch_model is not None:
        # Torch path gives us both prediction and Grad-CAM in one pass.
        result, gradcam_b64 = _predict_and_gradcam_torch(model_bundle, image)
        backend = "torch"
        if model_bundle.onnx_session is not None:
            note = "Prediction served by torch backend; ONNX backend is also loaded but unused for this request."
    elif model_bundle.onnx_session is not None:
        result = _predict_onnx(model_bundle, image)
        backend = "onnx"
        note = "Grad-CAM unavailable: only an ONNX backend is loaded. See TODO(phase 4) in serving/app.py."
    else:  # pragma: no cover - guarded by is_ready check above
        raise HTTPException(status_code=503, detail="No model backend available.")

    # TODO(phase 4): replace these placeholders with real calibrated
    # prediction intervals (e.g. via quantile heads or conformal prediction).
    return PredictionResponse(
        year=result["year"],
        year_confidence_interval=None,
        price_gbp=result["price_gbp"],
        price_confidence_interval=None,
        gradcam_png_base64=gradcam_b64,
        backend=backend,
        note=note,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
