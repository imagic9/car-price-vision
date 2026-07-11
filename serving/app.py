"""FastAPI demo server: upload a car photo, get predicted year + price plus
a Grad-CAM overlay.

Endpoints
---------
GET  /              -> serves static/index.html (product page)
GET  /health        -> {"status": "ok", "model_loaded": bool, "model_info": {...}?}
GET  /gallery       -> sample-photo gallery metadata (list of {file, brand, model,
                       true_year, true_price_gbp, bodytype, color}), or an empty
                       list if the gallery hasn't been staged on this server.
POST /predict       -> multipart image upload -> JSON prediction (+ base64 Grad-CAM PNG)
POST /predict-url   -> {"url": "..."} -> same prediction as /predict, but the image
                       is downloaded server-side first (see SSRF protections below).

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

SSRF protections (POST /predict-url)
-------------------------------------
/predict-url lets a visitor hand us a URL and have *this server* download
the image. That is a classic Server-Side Request Forgery (SSRF) primitive
if not locked down: without safeguards, an attacker could point it at
`http://169.254.169.254/...` (cloud metadata endpoints), `http://localhost:.../`,
or another container on this host's private Docker network (see
serving/deploy/docker-compose.yml's `car-net`) and use this server as a
proxy to reach them. `validate_image_url` / `_fetch_image_safely` below are
the sole gate against that and implement, in order:

  1. Scheme allow-list (http/https only); reject literal `localhost`.
  2. Resolve *every* A/AAAA record for the hostname via `socket.getaddrinfo`
     *before* any connection is attempted, and reject if *any* resolved
     address is private / loopback / link-local / multicast / reserved /
     unspecified (stdlib `ipaddress` checks). Also reject raw-IP URLs that
     point directly at such ranges.
  3. Redirects are not auto-followed by the HTTP client; we inspect each
     redirect ourselves, re-run the full validation above on the target,
     and refuse to hop to a different host, capping at 3 hops total.
  4. A hard 10 MB cap on the response body (checked against Content-Length
     up front, and against actual bytes streamed, in case the header lies
     or is absent) plus a 10s connect+read timeout.
  5. Content sniffing: whatever bytes we did download must decode as an
     image via Pillow, or the request is rejected with a 400.

Known residual limitation: step 2 resolves DNS once, immediately before
connecting, and the HTTP client performs its own resolution when it
connects (classic TOCTOU/DNS-rebinding window). Fully closing that gap
means connecting to the pre-resolved IP directly (pinning) while keeping
the original Host/SNI, which needs a custom transport; deliberately not
implemented here to keep this demo's serving code readable, but flagged
here for anyone hardening this beyond a course project.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import logging
import os
import socket
from pathlib import Path
from typing import Optional

import httpx
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

# Sample-photo gallery (frontend's "Gallery" tab). Staged separately from
# this repo (~150 images) -- see GET /gallery below for the graceful
# "not staged yet" fallback.
GALLERY_DIR = STATIC_DIR / "gallery"
GALLERY_JSON_PATH = GALLERY_DIR / "gallery.json"

# --- /predict-url SSRF protections (see module docstring) ------------------
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB cap on downloaded image size.
URL_FETCH_TIMEOUT_SECONDS = 10.0  # connect+read timeout, total.
MAX_URL_REDIRECTS = 3  # same-host redirects only; see _fetch_image_safely.


class PredictionResponse(BaseModel):
    year: int
    year_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    price_gbp: float
    price_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    gradcam_png_base64: Optional[str] = None
    backend: str
    note: Optional[str] = None


class PredictURLRequest(BaseModel):
    url: str


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
        # Raw parsed model_meta.json, kept around (beyond the fields already
        # unpacked above) so /health can surface e.g. a "metrics" key for the
        # frontend's "typical error" lines, without this class needing to
        # know every possible field up front.
        self.raw_meta: dict | None = None
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
            self.raw_meta = meta
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
    payload: dict = {"status": "ok", "model_loaded": ready}

    # Optional model_info block, built tolerantly from model_meta.json (any
    # of these fields -- including the whole file -- may be absent; the
    # frontend falls back to hardcoded README metrics if "metrics" isn't
    # here).
    meta = model_bundle.raw_meta if model_bundle is not None else None
    if meta:
        model_info = {}
        for key in ("backbone", "img_size", "metrics"):
            if key in meta:
                model_info[key] = meta[key]
        if model_info:
            payload["model_info"] = model_info

    return JSONResponse(payload)


@app.get("/gallery")
def gallery() -> JSONResponse:
    """Sample-photo gallery for the frontend's "Gallery" tab: pre-selected
    car photos with known ground truth (true_year, true_price_gbp, ...) so
    visitors can try a prediction without hunting for their own photo, and
    then judge accuracy against the real answer.

    The ~150 sample images are staged onto the server separately from this
    repo (they're not committed here); this endpoint must degrade
    gracefully -- never fabricate entries -- if that staging hasn't
    happened yet, so the frontend can hide the Gallery tab instead of
    showing a broken grid.
    """
    if not GALLERY_JSON_PATH.exists():
        return JSONResponse({"images": [], "note": "Gallery not staged on this server yet."})
    try:
        with GALLERY_JSON_PATH.open() as f:
            images = json.load(f)
    except Exception:
        logger.exception("Found %s but failed to parse it as JSON.", GALLERY_JSON_PATH)
        return JSONResponse({"images": [], "note": "Gallery metadata could not be read."})
    return JSONResponse({"images": images, "note": None})


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


def _ensure_model_ready() -> ModelBundle:
    """Shared readiness gate for /predict and /predict-url. Raises the same
    503s either endpoint would raise on its own; returns the bundle so
    callers don't need a second `model_bundle is None` check.
    """
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
    return model_bundle


def _run_prediction(bundle: ModelBundle, image: Image.Image) -> PredictionResponse:
    """Shared prediction path for /predict and /predict-url: pick a backend,
    run inference (+ Grad-CAM if torch is available), de-standardize, and
    build the response. Assumes `bundle` has already passed
    `_ensure_model_ready`.
    """
    gradcam_b64 = None
    note = None

    if bundle.torch_model is not None:
        # Torch path gives us both prediction and Grad-CAM in one pass.
        (year_z, log_price_z), gradcam_b64 = _predict_and_gradcam_torch(bundle, image)
        backend = "torch"
        if bundle.onnx_session is not None:
            note = "Prediction served by torch backend; ONNX backend is also loaded but unused for this request."
    elif bundle.onnx_session is not None:
        year_z, log_price_z = _predict_onnx(bundle, image)
        backend = "onnx"
        note = "Grad-CAM unavailable: only an ONNX backend is loaded. See TODO(phase 4) in serving/app.py."
    else:  # pragma: no cover - guarded by _ensure_model_ready's is_ready check
        raise HTTPException(status_code=503, detail="No model backend available.")

    real = _destandardize(year_z, log_price_z, bundle.target_norm)

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


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    bundle = _ensure_model_ready()

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file as an image: {exc}") from exc

    return _run_prediction(bundle, image)


class URLValidationError(ValueError):
    """Raised for any URL that fails the SSRF safety checks below. Its
    message is written to be safe to return to the client directly (no
    internal state/exception details).
    """


def _is_disallowed_ip(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    """True if `ip` must never be connected to from this server: private,
    loopback, link-local (this covers the 169.254.169.254 cloud-metadata
    address), multicast, reserved, or unspecified (0.0.0.0 / ::).
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_and_validate_host(hostname: str) -> list[str]:
    """Resolve every A/AAAA record for `hostname` and reject the host if
    *any* of them falls in a disallowed range (see _is_disallowed_ip) --
    not just the first one returned. Raises URLValidationError on any
    resolution failure or disallowed address.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise URLValidationError(f"Could not resolve host: {hostname}") from exc

    addresses: list[str] = []
    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip_str = sockaddr[0].split("%", 1)[0]  # strip IPv6 zone id (e.g. "fe80::1%eth0")
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_disallowed_ip(ip):
            raise URLValidationError("That URL resolves to a disallowed address; only public hosts are allowed.")
        addresses.append(ip_str)

    if not addresses:
        raise URLValidationError(f"Could not resolve any address for host: {hostname}")
    return addresses


def validate_image_url(url: str) -> tuple[str, list[str]]:
    """Validate a user-supplied image URL *before* any network request is
    made to it. Returns (hostname, resolved_ip_addresses) on success;
    raises URLValidationError otherwise.

    SECURITY-CRITICAL: this is the sole gate keeping /predict-url from
    being usable as an SSRF primitive against internal services (cloud
    metadata endpoints, localhost, other containers on this host's private
    network -- see the module docstring). Do not weaken these checks.
    """
    try:
        parsed = httpx.URL(url)
    except Exception as exc:
        raise URLValidationError("That doesn't look like a valid URL.") from exc

    if parsed.scheme not in ("http", "https"):
        raise URLValidationError(f"Unsupported URL scheme ({parsed.scheme or 'none'}); only http/https are allowed.")

    hostname = parsed.host
    if not hostname:
        raise URLValidationError("URL has no hostname.")
    if hostname.lower() == "localhost":
        raise URLValidationError("URLs pointing at localhost are not allowed.")

    # A literal IP in the URL still has to pass the same disallow-list.
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and _is_disallowed_ip(literal_ip):
        raise URLValidationError("That URL points at a disallowed address.")

    resolved = _resolve_and_validate_host(hostname)
    return hostname, resolved


def _fetch_image_safely(url: str) -> bytes:
    """Download image bytes from a user-supplied URL, enforcing every
    protection described in the module docstring's SSRF section: validated
    scheme/host/resolved-IPs before connecting, same-host-only redirects
    capped at MAX_URL_REDIRECTS, a hard MAX_IMAGE_BYTES cap (checked against
    both Content-Length and actual streamed bytes), and a
    URL_FETCH_TIMEOUT_SECONDS connect+read timeout.

    Raises URLValidationError (message is safe to show the client) on any
    problem; never lets a raw exception/traceback reach the caller.
    """
    current_url = url
    for _hop in range(MAX_URL_REDIRECTS + 1):
        hostname, _resolved_ips = validate_image_url(current_url)
        try:
            with httpx.Client(follow_redirects=False, timeout=URL_FETCH_TIMEOUT_SECONDS) as client:
                with client.stream("GET", current_url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise URLValidationError("The server redirected without a target.")
                        next_url = httpx.URL(current_url).join(location)
                        if next_url.host != hostname:
                            raise URLValidationError("Redirects to a different host are not allowed.")
                        current_url = str(next_url)
                        continue

                    if response.status_code >= 400:
                        raise URLValidationError(f"The server returned an error (HTTP {response.status_code}).")

                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            if int(content_length) > MAX_IMAGE_BYTES:
                                raise URLValidationError("Image exceeds the 10 MB size limit.")
                        except ValueError:
                            pass  # malformed header; fall through to the streamed-size check below

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            raise URLValidationError("Image exceeds the 10 MB size limit.")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except httpx.HTTPError as exc:
            raise URLValidationError("Could not download the image from that URL.") from exc

    raise URLValidationError("Too many redirects.")


@app.post("/predict-url", response_model=PredictionResponse)
async def predict_url(payload: PredictURLRequest) -> PredictionResponse:
    # SSRF validation runs *before* the model-readiness check: it's cheap,
    # doesn't need the model, and a malformed/unsafe URL should be rejected
    # the same way regardless of whether a model happens to be loaded.
    try:
        image_bytes = _fetch_image_safely(payload.url)
    except URLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = _ensure_model_ready()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # never echo the underlying decoder exception
        raise HTTPException(status_code=400, detail="Downloaded content is not a valid image.") from exc

    return _run_prediction(bundle, image)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
