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

De-standardization
-------------------
Both model backends output predictions in z-space (see
data/dataset.py's `target_norm` standardization and train.py); this app
never sees real years/GBP directly out of the model. On startup it loads
`model_meta.json` (written by scripts/export_onnx.py) which is the single
source of truth for `target_norm` (plus img_size/backbone/normalization
stats), and de-standardizes every prediction before returning it:

    real_year      = year_z * target_norm["year_std"] + target_norm["year_mean"]
    real_log_price = log_price_z * target_norm["log_price_std"] + target_norm["log_price_mean"]
    real_price_gbp = exp(real_log_price)

If `model_meta.json` is missing, `target_norm` (and backbone/img_size) are
recovered from the torch checkpoint instead (train.py's save_checkpoint
writes these into every checkpoint). Norm constants are never hardcoded
here.

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
import json
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
# Written by scripts/export_onnx.py alongside model.onnx/model.pt; the
# single source of truth for target_norm + img_size + backbone. Falls back
# to the torch checkpoint's own fields if this file isn't present.
MODEL_META_PATH = Path(os.environ.get("CPV_MODEL_META", "/models/model_meta.json"))
# Last-resort fallbacks only used if neither model_meta.json nor the torch
# checkpoint carry backbone/img_size (e.g. a very old checkpoint).
BACKBONE_NAME_FALLBACK = os.environ.get("CPV_BACKBONE", "convnext_tiny")
IMG_SIZE_FALLBACK = int(os.environ.get("CPV_IMG_SIZE", "224"))


class PredictionResponse(BaseModel):
    year: int
    year_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    price_gbp: float
    price_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    gradcam_png_base64: Optional[str] = None
    backend: str
    note: Optional[str] = None


class ModelBundle:
    """Lazily loads and holds whichever model backend(s) are available, plus
    the de-standardization/preprocessing metadata (`target_norm`, `img_size`,
    `backbone_name`) needed to turn raw model outputs into real predictions.
    """

    def __init__(self) -> None:
        self.onnx_session = None
        self.torch_model = None
        self.gradcam_target_layer = None
        self.target_norm: dict | None = None
        self.img_size: int = IMG_SIZE_FALLBACK
        self.backbone_name: str = BACKBONE_NAME_FALLBACK
        self._load()

    def _load_meta(self) -> dict | None:
        if not MODEL_META_PATH.exists():
            logger.warning(
                "No model_meta.json found at %s (set CPV_MODEL_META). Will fall back to reading "
                "target_norm/backbone/img_size from the torch checkpoint if one is available.",
                MODEL_META_PATH,
            )
            return None
        try:
            with MODEL_META_PATH.open() as f:
                meta = json.load(f)
            logger.info("Loaded model_meta.json from %s", MODEL_META_PATH)
            return meta
        except Exception:
            logger.exception("Found %s but failed to parse it as JSON.", MODEL_META_PATH)
            return None

    def _load(self) -> None:
        meta = self._load_meta()
        if meta is not None:
            self.target_norm = meta.get("target_norm")
            self.img_size = int(meta.get("img_size", self.img_size))
            self.backbone_name = meta.get("backbone", self.backbone_name)

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

                checkpoint = torch.load(TORCH_CHECKPOINT_PATH, map_location="cpu")

                if self.target_norm is None:
                    self.target_norm = checkpoint.get("target_norm")
                    if self.target_norm is not None:
                        logger.info("target_norm not found in model_meta.json; recovered from the torch checkpoint.")
                if checkpoint.get("backbone_name"):
                    self.backbone_name = checkpoint["backbone_name"]
                if meta is None and checkpoint.get("img_size"):
                    self.img_size = int(checkpoint["img_size"])

                model = MultiTaskCarNet(backbone_name=self.backbone_name, pretrained=False)
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

        if self.is_ready and self.target_norm is None:
            logger.error(
                "A model backend loaded but target_norm is unavailable (missing from both model_meta.json "
                "and the torch checkpoint). Predictions cannot be de-standardized; /predict will fail."
            )

    @property
    def is_ready(self) -> bool:
        return self.onnx_session is not None or self.torch_model is not None

    @property
    def can_gradcam(self) -> bool:
        return self.torch_model is not None

    @property
    def can_predict(self) -> bool:
        """Ready to serve /predict: a backend is loaded AND we have the
        target_norm constants needed to de-standardize its output.
        """
        return self.is_ready and self.target_norm is not None


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


def _preprocess(image: Image.Image, img_size: int) -> np.ndarray:
    """Resize/normalize a PIL image into a (1, 3, H, W) float32 array using
    the same ImageNet stats as data/transforms.py:eval_transforms.
    """
    from car_price_vision.data.transforms import eval_transforms

    tensor = eval_transforms(img_size)(image)
    return tensor.unsqueeze(0).numpy()


def _destandardize(year_z: float, log_price_z: float, target_norm: dict) -> dict:
    """Invert the z-score standardization applied to targets at train time
    (see data/dataset.py). Returns real year (float, not yet rounded) and
    real price in GBP.
    """
    year = year_z * target_norm["year_std"] + target_norm["year_mean"]
    log_price = log_price_z * target_norm["log_price_std"] + target_norm["log_price_mean"]
    price_gbp = float(np.exp(log_price))
    return {"year": float(year), "price_gbp": price_gbp}


def _predict_onnx(bundle: ModelBundle, image: Image.Image) -> tuple[float, float]:
    """Returns (year_z, log_price_z) -- still in z-space, see model_meta.json's
    "outputs" field which documents this exact order: [year_z, log_price_z].
    """
    input_array = _preprocess(image, bundle.img_size)
    input_name = bundle.onnx_session.get_inputs()[0].name
    outputs = bundle.onnx_session.run(None, {input_name: input_array.astype(np.float32)})
    return float(outputs[0].squeeze()), float(outputs[1].squeeze())


def _predict_and_gradcam_torch(bundle: ModelBundle, image: Image.Image) -> tuple[tuple[float, float], Optional[str]]:
    """Returns ((year_z, log_price_z), gradcam_png_base64_or_none)."""
    import torch

    from car_price_vision.interpret.gradcam import GradCAM, overlay_heatmap

    tensor = torch.from_numpy(_preprocess(image, bundle.img_size))

    with torch.no_grad():
        preds = bundle.torch_model(tensor)
    year_z = float(preds["year"].item())
    log_price_z = float(preds["log_price"].item())

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

    return (year_z, log_price_z), gradcam_b64


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
    if model_bundle.target_norm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "A model backend is loaded but target_norm (de-standardization constants) is "
                f"unavailable. Provide {MODEL_META_PATH} (see scripts/export_onnx.py) or a torch "
                "checkpoint that includes a target_norm field."
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
        (year_z, log_price_z), gradcam_b64 = _predict_and_gradcam_torch(model_bundle, image)
        backend = "torch"
        if model_bundle.onnx_session is not None:
            note = "Prediction served by torch backend; ONNX backend is also loaded but unused for this request."
    elif model_bundle.onnx_session is not None:
        year_z, log_price_z = _predict_onnx(model_bundle, image)
        backend = "onnx"
        note = "Grad-CAM unavailable: only an ONNX backend is loaded. See TODO(phase 4) in serving/app.py."
    else:  # pragma: no cover - guarded by is_ready check above
        raise HTTPException(status_code=503, detail="No model backend available.")

    real = _destandardize(year_z, log_price_z, model_bundle.target_norm)

    # TODO(phase 4): replace these placeholders with real calibrated
    # prediction intervals (e.g. via quantile heads or conformal prediction).
    return PredictionResponse(
        year=int(round(real["year"])),
        year_confidence_interval=None,
        price_gbp=round(real["price_gbp"] / 10.0) * 10.0,
        price_confidence_interval=None,
        gradcam_png_base64=gradcam_b64,
        backend=backend,
        note=note,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
