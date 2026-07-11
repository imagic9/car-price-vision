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
POST /predict-url   -> {"url": "..."} -> JSON prediction, but the content is
                       downloaded server-side first (see SSRF protections below).
                       Two shapes are accepted for `url`:
                         1. A direct image URL -> same single-photo prediction
                            as /predict (photos_used=1, page fields null).
                         2. An advert/listing page URL (auto.ria.com, mobile.de,
                            autoscout24, ...) -> the page's HTML is fetched
                            (same SSRF gate, separate 3 MB cap), candidate photo
                            URLs are extracted from it (see "Advert-page photo
                            extraction" below), each is fetched through the
                            *same* SSRF gate as a direct image URL, and the
                            response is the MEDIAN of the per-photo predictions
                            (see "Multi-photo aggregation" below). Which shape
                            was given is auto-detected after download by
                            content sniffing -- never by trusting the URL's
                            shape or the response's Content-Type alone.
                            For this advert-page shape only, the response
                            additionally carries the advert's OWN year and
                            asking price when they can be confidently read
                            off the page (advert_year, advert_price_original,
                            advert_price_gbp, rate_source -- see "Advert
                            fact extraction" below), so the UI can show a
                            predicted-vs-advertised comparison.

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

Advert-page photo extraction (POST /predict-url, page-URL shape)
------------------------------------------------------------------
Real visitors usually paste a marketplace *listing* URL, not a direct
image link. If the downloaded body doesn't decode as an image and looks
like HTML (Content-Type says so, or the body itself sniffs as HTML --
see `_looks_like_html`), `extract_photo_urls_from_html` pulls candidate
photo URLs out of the raw markup with stdlib-only parsing (`html.parser`
+ `json.loads` on embedded JSON-LD -- no BeautifulSoup), in priority
order:

  1. `<meta property="og:image">` / `og:image:secure_url`
  2. `<meta name="twitter:image">`
  3. `application/ld+json` `<script>` blocks' `"image"` field (string or list)
  4. `<img src>` / `<img data-src>`, as a last resort

Relative URLs are resolved against the page's *final* URL (after
redirects); only http/https survive; obvious decorative images (URL
contains "logo"/"icon"/"sprite"/"avatar") are dropped; results are
de-duplicated and capped at MAX_PHOTOS_PER_PAGE (8).

SECURITY: every extracted URL is a candidate SSRF target, because it came
from attacker-controlled page content, not from the user directly. Each
one is fetched through the *exact same* `validate_image_url` +
`_fetch_image_safely` gate documented above -- an advert page cannot use
its <img> tags to reach a host the direct-URL path would have refused.
Photos that fail to fetch/decode/predict are skipped, not fatal; if none
survive, the endpoint returns 422.

Multi-photo aggregation
------------------------
Each surviving photo is run through the model to get a (year_z,
log_price_z) pair in z-space (see `_predict_z_only`). The response is the
per-field MEDIAN of those z-space values, de-standardized once at the end
(`_aggregate_photo_predictions`) -- median rather than mean because
listing photos routinely include interior/dashboard/engine-bay shots that
are out-of-distribution for this exterior-trained model; a mean would let
a minority of such shots drag the estimate, while the median shrugs off
up to roughly half of them. Grad-CAM is computed for only one photo, the
"representative" one whose z-vector is closest (L2) to the median vector
-- computing it for every photo would multiply the (CPU-only) Grad-CAM
cost by the photo count for no benefit, since only one overlay is shown.

Advert fact extraction (POST /predict-url, page-URL shape)
-----------------------------------------------------------
The same advert page that supplied the photos usually also states the
car's year and asking price -- the most interesting "ground truth" a
visitor could compare the model against. `extract_advert_facts_from_html`
reads them off the already-downloaded HTML (the page is parsed exactly
once per request -- `_parse_advert_page` -- and the resulting parse is
shared with the photo extraction above). Priority order per field,
stopping at the first confident hit:

  1. JSON-LD (`application/ld+json`) nodes of schema.org type
     Car/Vehicle/Product: price from `offers.price` + `offers.priceCurrency`
     (`offers` may be a dict or a list; a top-level `price` also counts),
     year from `vehicleModelDate` / `productionDate` / `releaseDate` /
     `dateVehicleFirstRegistered`, or a 4-digit year in the node's `name`.
  2. Meta/OpenGraph: a 4-digit year (1950..2029) in `og:title` or
     `<title>` -- the LAST match in the string, since model names routinely
     contain digits ("Nissan 350Z 2008"); price from
     `product:price:amount` + `product:price:currency` metas.
  3. Conservative raw-text fallback, price only: the currency-marked
     amount nearest the top of the document ($..., ...грн/₴, €..., £...,
     or "... USD/EUR/UAH/GBP"), with thousand separators normalized.

Honesty rule: a field that can't be confidently extracted stays None and
the corresponding response fields are simply omitted -- never guessed.
Sanity bounds reject nonsense (price outside 50..5,000,000, year outside
1950..2029). The advert's price is also converted to GBP (the model's
output currency) via serving/currency.py -- live rates from a fixed,
non-user-controlled API with a 24h in-memory cache and a hardcoded
approximate fallback snapshot; `rate_source` in the response says which
was used ("live"/"fallback"). Any failure anywhere in fact extraction or
conversion degrades to "no comparison shown", never to a failed
prediction (see `_attach_advert_facts`).
"""

from __future__ import annotations

import base64
import html.parser
import io
import ipaddress
import json
import logging
import os
import re
import socket
import statistics
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# serving/ has no __init__.py: inside the serving container it runs as a
# top-level module (`uvicorn app:app` with WORKDIR /app/serving -- see
# serving/Dockerfile), while tests and `uvicorn serving.app:app` from the
# repo root import it as a namespace-package member. Support both shapes.
try:
    from serving import currency
except ImportError:  # running from inside serving/ as a top-level module
    import currency

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
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB cap on a downloaded direct-image body.
MAX_HTML_BYTES = 3 * 1024 * 1024  # 3 MB cap on a downloaded advert-page HTML body.
URL_FETCH_TIMEOUT_SECONDS = 10.0  # connect+read timeout, total.
MAX_URL_REDIRECTS = 3  # same-host redirects only; see _fetch_bytes_safely.

# --- Advert-page photo extraction (see module docstring) --------------------
MAX_PHOTOS_PER_PAGE = 8  # cap on candidate photo URLs pulled out of one page.
# Cheap heuristic to skip obvious decorative/non-car images (site logos,
# nav icons, sprite sheets, user-avatar placeholders) -- a substring match
# on the resolved URL, kept intentionally simple rather than trying to be
# a general image classifier.
_ICON_URL_KEYWORDS = ("logo", "icon", "sprite", "avatar")

# --- Advert fact extraction (see module docstring) ---------------------------
# Sanity bounds -- outside these, an extracted value is treated as noise
# (a phone number, a mileage figure, a stray SKU) and dropped, per the
# "never guess" rule.
ADVERT_YEAR_MIN = 1950
ADVERT_YEAR_MAX = 2029
ADVERT_PRICE_MIN = 50.0
ADVERT_PRICE_MAX = 5_000_000.0
# 4-digit year in ADVERT_YEAR_MIN..ADVERT_YEAR_MAX -- the range is baked
# into the regex itself so every match is already in-bounds.
_ADVERT_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
# schema.org types whose year/price fields we trust (lowercased; @type may
# also carry a full URL like "https://schema.org/Car", hence the suffix
# match in _jsonld_node_is_advert_type).
_JSONLD_ADVERT_TYPES = ("car", "vehicle", "product")
_JSONLD_YEAR_KEYS = ("vehicleModelDate", "productionDate", "releaseDate", "dateVehicleFirstRegistered")
# Raw-text price fallback: currency-marked amounts only (a bare number is
# never trusted). Amount character classes deliberately use a literal
# space + NBSP rather than \s so a match can't glue digits together across
# newlines. The UAH/EUR/GBP/code patterns allow "." and "," as thousand
# separators (see _parse_advert_amount); the mapping $->USD, грн/₴->UAH,
# €->EUR, £->GBP is encoded per-pattern. Bounded quantifiers ({3,12}) keep
# a match from swallowing arbitrarily long digit runs.
_TEXT_PRICE_PATTERNS: list[tuple[re.Pattern[str], Optional[str]]] = [
    (re.compile(r"\$\s?([\d  ,]{1,12}\d)"), "USD"),
    (re.compile(r"(\d[\d  ,]{0,11})\s?(?:грн|₴)"), "UAH"),
    (re.compile(r"€\s?([\d  .,]{1,12}\d)"), "EUR"),
    (re.compile(r"£\s?([\d  .,]{1,12}\d)"), "GBP"),
    (re.compile(r"(\d[\d  .,]{0,11})\s?(USD|EUR|UAH|GBP)\b"), None),  # currency read from group 2
]


class PerPhotoPrediction(BaseModel):
    """One entry of PredictionResponse.per_photo -- the per-photo
    (pre-aggregation) prediction for one photo extracted from an advert
    page. See _run_prediction_for_photos.
    """

    url: str
    year: int
    price_gbp: float


class AdvertPriceOriginal(BaseModel):
    """The advert page's own asking price, verbatim in the page's currency
    (before any GBP conversion) -- see PredictionResponse.advert_price_gbp
    for the converted figure and extract_advert_facts_from_html for how
    it's read off the page.
    """

    amount: float
    currency: str  # ISO 4217, e.g. "USD" / "EUR" / "UAH" / "GBP"


class PredictionResponse(BaseModel):
    year: int
    year_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    price_gbp: float
    price_confidence_interval: Optional[list[float]] = None  # TODO(phase 4): placeholder, not yet calibrated
    gradcam_png_base64: Optional[str] = None
    backend: str
    note: Optional[str] = None
    # --- Advert-page multi-photo aggregation (POST /predict-url only; see
    # extract_photo_urls_from_html / _run_prediction_for_photos). All
    # optional/defaulted so /predict and direct-image /predict-url requests
    # are unaffected: they keep photos_used=1 and every page field null. ---
    photos_used: int = 1
    photos_found: Optional[int] = None
    page_url: Optional[str] = None
    per_photo: Optional[list[PerPhotoPrediction]] = None
    representative_photo_url: Optional[str] = None
    # --- Advert-page "own figures" comparison (POST /predict-url, page-URL
    # shape only; see extract_advert_facts_from_html / _attach_advert_facts).
    # All optional so every other request shape is unaffected; any of them
    # may independently be null when the page didn't state that fact
    # confidently enough (the "never guess" rule). ---
    advert_year: Optional[int] = None
    advert_price_original: Optional[AdvertPriceOriginal] = None
    advert_price_gbp: Optional[float] = None
    rate_source: Optional[str] = None  # "live" | "fallback"; see serving/currency.py


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


def _predict_z_only(bundle: ModelBundle, image: Image.Image) -> tuple[float, float]:
    """Returns (year_z, log_price_z) using whichever backend is loaded,
    *without* computing Grad-CAM even if the torch backend is available.

    Used for the per-photo pass over an advert page's multiple photos
    (see _run_prediction_for_photos): running the Grad-CAM backward pass
    for every one of up to MAX_PHOTOS_PER_PAGE photos would multiply a
    CPU-only cost for no benefit, since only the single "representative"
    photo's overlay is ever shown. Prefers ONNX (faster) when both
    backends are loaded, matching _run_prediction_for_photos' reported
    `backend` field.
    """
    if bundle.onnx_session is not None:
        return _predict_onnx(bundle, image)

    import torch

    tensor = torch.from_numpy(_preprocess(image, bundle.img_size))
    with torch.no_grad():
        preds = bundle.torch_model(tensor)
    return float(preds["year"].item()), float(preds["log_price"].item())


def _aggregate_photo_predictions(z_vectors: list[tuple[float, float]]) -> tuple[float, float, int]:
    """Aggregate per-photo (year_z, log_price_z) tuples from multiple
    advert-page photos into one (median_year_z, median_log_price_z,
    representative_index).

    Median, not mean: listing pages routinely surface interior/dashboard/
    engine-bay/detail shots alongside exterior ones, and this model is
    trained on (and calibrated for) exterior shots -- those other photos'
    predictions are out-of-distribution outliers in z-space. A mean would
    let a minority of such shots drag the aggregate; per-field medians are
    robust to up to roughly half the photos being such outliers.

    `representative_index` is the index of the z-vector with the smallest
    L2 distance to the (median_year_z, median_log_price_z) point -- i.e.
    the *actual* photo that best represents the aggregate result, used to
    pick which single photo gets a Grad-CAM overlay and is shown in the UI.

    Pure function (no I/O, no model) so it's directly unit-testable; see
    tests/test_serving_page_extract.py.
    """
    year_zs = [z[0] for z in z_vectors]
    log_price_zs = [z[1] for z in z_vectors]
    median_year_z = statistics.median(year_zs)
    median_log_price_z = statistics.median(log_price_zs)

    representative_index = min(
        range(len(z_vectors)),
        key=lambda i: (z_vectors[i][0] - median_year_z) ** 2 + (z_vectors[i][1] - median_log_price_z) ** 2,
    )
    return median_year_z, median_log_price_z, representative_index


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


def _fetch_bytes_safely(url: str) -> tuple[bytes, str, str]:
    """Download bytes from a user-supplied (or advert-page-extracted) URL,
    enforcing every protection described in the module docstring's SSRF
    section: validated scheme/host/resolved-IPs before connecting,
    same-host-only redirects capped at MAX_URL_REDIRECTS, and a
    URL_FETCH_TIMEOUT_SECONDS connect+read timeout.

    Shared by both direct-image and advert-page-HTML fetches (see the
    "Advert-page photo extraction" section of the module docstring), so the
    byte cap has to be picked per-request rather than being a single
    constant: once the response headers arrive (but *before* any body bytes
    are read), the advertised Content-Type decides whether MAX_HTML_BYTES
    (3 MB, for `text/html`-ish responses) or MAX_IMAGE_BYTES (10 MB,
    everything else -- including images and any response with a missing or
    generic Content-Type) applies, and that cap is then enforced against
    both Content-Length and actual streamed bytes, exactly as before this
    function had a single fixed cap. A response that lies about its
    Content-Type only ever gets the *larger* of the two caps, never an
    unbounded one, so this can't be used to bypass size limits -- it can at
    most let some HTML through under the image cap instead of the HTML cap,
    which the caller's post-download content sniffing (`_looks_like_html`)
    still catches correctly.

    Returns (body_bytes, content_type_header_lowercased, final_url_after_redirects).
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

                    content_type = response.headers.get("content-type", "").lower()
                    max_bytes = MAX_HTML_BYTES if "html" in content_type else MAX_IMAGE_BYTES
                    limit_mb = max_bytes // (1024 * 1024)

                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            if int(content_length) > max_bytes:
                                raise URLValidationError(f"Content exceeds the {limit_mb} MB size limit.")
                        except ValueError:
                            pass  # malformed header; fall through to the streamed-size check below

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise URLValidationError(f"Content exceeds the {limit_mb} MB size limit.")
                        chunks.append(chunk)
                    return b"".join(chunks), content_type, current_url
        except httpx.HTTPError as exc:
            raise URLValidationError("Could not download content from that URL.") from exc

    raise URLValidationError("Too many redirects.")


def _fetch_image_safely(url: str) -> bytes:
    """Thin wrapper over _fetch_bytes_safely for callers that only ever
    expect an image (single direct-image /predict-url requests, and each
    individual photo extracted from an advert page) and don't need the
    Content-Type / final-URL that HTML handling requires.
    """
    body, _content_type, _final_url = _fetch_bytes_safely(url)
    return body


def _try_decode_image(body: bytes) -> Optional[Image.Image]:
    """Best-effort image decode; returns None (never raises) so the caller
    can fall through to advert-page HTML handling instead of erroring out.
    """
    try:
        return Image.open(io.BytesIO(body)).convert("RGB")
    except Exception:
        return None


def _looks_like_html(body: bytes, content_type: str) -> bool:
    """Best-effort sniff used only to decide *how to interpret* a downloaded
    body that already failed to decode as an image -- this never bypasses
    or weakens any SSRF check (those already happened in
    _fetch_bytes_safely before a single byte was read). Trusts the
    Content-Type header first; falls back to a cheap substring sniff of the
    first few KB in case the header is missing or generic (e.g.
    "application/octet-stream"), which is common for misconfigured static
    hosting of listing pages.
    """
    if "html" in content_type:
        return True
    head = body[:4096].lower()
    return b"<html" in head or b"<!doctype html" in head


class _AdvertPageParser(html.parser.HTMLParser):
    """Collects candidate photo URLs *and* advert-fact raw material (JSON-LD
    nodes, og:title / <title>, price metas) from an advert/listing page's
    raw HTML in a single pass, stdlib-only (no BeautifulSoup) -- see
    extract_photo_urls_from_html and extract_advert_facts_from_html for how
    the collected pieces are combined/prioritized. `html.parser` itself
    already tolerates malformed/unclosed markup (it does not raise on bad
    HTML the way a strict XML parser would), so this class doesn't need its
    own recovery logic beyond guarding the embedded-JSON-LD parse.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.og_images: list[str] = []
        self.twitter_images: list[str] = []
        self.ldjson_images: list[str] = []
        self.img_srcs: list[str] = []
        # Advert-fact raw material (see extract_advert_facts_from_html).
        self.ldjson_nodes: list[dict] = []  # every parsed JSON-LD object, in document order
        self.og_title: Optional[str] = None
        self.title: Optional[str] = None
        self.meta_price_amount: Optional[str] = None
        self.meta_price_currency: Optional[str] = None
        self._in_ldjson_script = False
        self._ldjson_buffer: list[str] = []
        self._in_title = False
        self._title_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_d = {k.lower(): v for k, v in attrs if k is not None}
        if tag == "meta":
            prop = (attrs_d.get("property") or attrs_d.get("name") or "").lower()
            content = attrs_d.get("content")
            if content:
                if prop in ("og:image", "og:image:secure_url"):
                    self.og_images.append(content)
                elif prop in ("twitter:image", "twitter:image:src"):
                    self.twitter_images.append(content)
                elif prop == "og:title" and self.og_title is None:
                    self.og_title = content
                elif prop in ("product:price:amount", "og:price:amount") and self.meta_price_amount is None:
                    self.meta_price_amount = content
                elif prop in ("product:price:currency", "og:price:currency") and self.meta_price_currency is None:
                    self.meta_price_currency = content
        elif tag == "script":
            script_type = (attrs_d.get("type") or "").lower()
            self._in_ldjson_script = script_type == "application/ld+json"
            if self._in_ldjson_script:
                self._ldjson_buffer = []
        elif tag == "title":
            self._in_title = True
            self._title_buffer = []
        elif tag == "img":
            src = attrs_d.get("src") or attrs_d.get("data-src")
            if src:
                self.img_srcs.append(src)

    def handle_data(self, data: str) -> None:
        if self._in_ldjson_script:
            self._ldjson_buffer.append(data)
        if self._in_title:
            self._title_buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_ldjson_script:
            self._in_ldjson_script = False
            self._ingest_ldjson("".join(self._ldjson_buffer))
            self._ldjson_buffer = []
        elif tag == "title" and self._in_title:
            self._in_title = False
            if self.title is None:
                self.title = "".join(self._title_buffer).strip() or None
            self._title_buffer = []

    def _ingest_ldjson(self, raw_json: str) -> None:
        """Parse one JSON-LD <script> block: remember every object node (for
        extract_advert_facts_from_html) and collect its "image" field(s)
        (for extract_photo_urls_from_html). One level of "@graph" is
        flattened -- a very common wrapper on real marketplace pages.
        """
        try:
            data = json.loads(raw_json)
        except (json.JSONDecodeError, ValueError):
            return  # malformed JSON-LD is common in the wild; just skip it
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            graph = node.get("@graph")
            subnodes = [node] + [g for g in graph if isinstance(g, dict)] if isinstance(graph, list) else [node]
            for sub in subnodes:
                self.ldjson_nodes.append(sub)
                image = sub.get("image")
                if isinstance(image, str):
                    self.ldjson_images.append(image)
                elif isinstance(image, list):
                    self.ldjson_images.extend(item for item in image if isinstance(item, str))


def _decode_html(html_content: bytes | str) -> str:
    """bytes -> str (utf-8, lossy) passthrough helper so the extraction
    functions below accept either shape.
    """
    if isinstance(html_content, bytes):
        return html_content.decode("utf-8", errors="ignore")
    return html_content


def _parse_advert_page(html_content: bytes | str) -> Optional[_AdvertPageParser]:
    """Run _AdvertPageParser over the page once and return it (or None if
    even the tolerant stdlib parser blew up). The endpoint parses each
    downloaded page exactly once and hands the result to both
    extract_photo_urls_from_html and extract_advert_facts_from_html via
    their `parser` parameter.
    """
    try:
        parser = _AdvertPageParser()
        parser.feed(_decode_html(html_content))
        parser.close()
        return parser
    except Exception:
        logger.exception("Failed to parse advert page HTML.")
        return None


def extract_photo_urls_from_html(
    html_bytes: bytes, base_url: str, parser: Optional[_AdvertPageParser] = None
) -> list[str]:
    """Extract candidate car-photo URLs from an advert/listing page's raw
    HTML (see the module docstring's "Advert-page photo extraction"
    section for the full rationale). Priority order, highest first:
    og:image/og:image:secure_url metas, twitter:image metas, JSON-LD
    "image" fields (string or list), and finally <img src>/<img data-src>
    as a last resort to fill out the list.

    Relative URLs are resolved against `base_url` (pass the *final* URL
    after redirects, not the original user-supplied one); only http/https
    survive; results are de-duplicated preserving order and capped at
    MAX_PHOTOS_PER_PAGE.

    Pass an already-built `parser` (from _parse_advert_page) to skip
    re-parsing the same page; omitted, the page is parsed here.

    Never raises: malformed HTML/JSON degrades to fewer (possibly zero)
    candidates rather than propagating an exception, so one broken listing
    page can't 500 the endpoint.
    """
    if parser is None:
        parser = _parse_advert_page(html_bytes)
    if parser is None:
        return []

    candidates = parser.og_images + parser.twitter_images + parser.ldjson_images + parser.img_srcs

    seen: set[str] = set()
    result: list[str] = []
    for raw_url in candidates:
        raw_url = raw_url.strip()
        if not raw_url:
            continue
        try:
            resolved = str(httpx.URL(base_url).join(raw_url))
        except Exception:
            continue  # malformed individual URL; skip, don't fail the whole page
        if httpx.URL(resolved).scheme not in ("http", "https"):
            continue
        lowered = resolved.lower()
        if any(keyword in lowered for keyword in _ICON_URL_KEYWORDS):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
        if len(result) >= MAX_PHOTOS_PER_PAGE:
            break
    return result


def _parse_advert_amount(raw: str) -> Optional[float]:
    """Parse a human-formatted money amount ("8 500", "1 250 000", "12.500",
    "12,999.99") into a float, or None if it doesn't look like one.

    Separator rules, deliberately simple: spaces/NBSPs are always thousand
    separators; a FINAL "." or "," followed by exactly 1-2 digits is a
    decimal point; every other "." / "," is a thousand separator (so
    "12.500" is twelve and a half thousand, the common European style on
    e.g. German/Ukrainian listings, not 12.5).
    """
    cleaned = raw.replace(" ", "").replace(" ", "").strip()
    if not cleaned:
        return None
    decimal_part = ""
    m = re.fullmatch(r"(.+)[.,](\d{1,2})", cleaned)
    if m:
        cleaned, decimal_part = m.group(1), m.group(2)
    integer_digits = cleaned.replace(",", "").replace(".", "")
    if not integer_digits.isdigit():
        return None
    return float(integer_digits + ("." + decimal_part if decimal_part else ""))


def _advert_price_in_bounds(price: Optional[float]) -> bool:
    """Sanity gate for every extracted price, whatever its source -- see
    ADVERT_PRICE_MIN/MAX and the "never guess" rule in the module docstring.
    """
    return price is not None and ADVERT_PRICE_MIN <= price <= ADVERT_PRICE_MAX


def _last_year_in_text(text: str) -> Optional[int]:
    """LAST in-range 4-digit year in `text`, or None. Last rather than
    first because model names routinely contain digit runs and the year
    conventionally trails the title ("Nissan 350Z 2008", "BMW 2002 Turbo
    1974"). Range 1950..2029 is baked into _ADVERT_YEAR_RE itself.
    """
    matches = _ADVERT_YEAR_RE.findall(text)
    return int(matches[-1]) if matches else None


def _jsonld_node_is_advert_type(node: dict) -> bool:
    """True if the JSON-LD node's @type (string or list, possibly a full
    schema.org URL) is one of _JSONLD_ADVERT_TYPES.
    """
    raw_type = node.get("@type")
    types = raw_type if isinstance(raw_type, list) else [raw_type]
    return any(
        isinstance(t, str) and t.strip().lower().rstrip("/").rsplit("/", 1)[-1] in _JSONLD_ADVERT_TYPES
        for t in types
    )


def _jsonld_year(node: dict) -> Optional[int]:
    """Year from a Car/Vehicle/Product JSON-LD node: the dedicated schema.org
    date fields first (values like 2016, "2016" or "2016-03-01" all work --
    the year is regex'd out of the stringified value), then a 4-digit year
    in the node's "name" as a fallback.
    """
    for key in _JSONLD_YEAR_KEYS:
        value = node.get(key)
        if value is None:
            continue
        match = _ADVERT_YEAR_RE.search(str(value))
        if match:
            return int(match.group(0))
    name = node.get("name")
    if isinstance(name, str):
        return _last_year_in_text(name)
    return None


def _jsonld_price(node: dict) -> tuple[Optional[float], Optional[str]]:
    """(price, currency) from a Car/Vehicle/Product JSON-LD node's offers
    (dict or list of dicts; `price` + `priceCurrency`), or from a `price`
    directly on the node. (None, None) if nothing in-bounds is found;
    currency may be None when the node states a price without one.
    """
    offers = node.get("offers")
    if isinstance(offers, dict):
        offer_candidates: list[dict] = [offers]
    elif isinstance(offers, list):
        offer_candidates = [o for o in offers if isinstance(o, dict)]
    else:
        offer_candidates = []

    for candidate in offer_candidates + [node]:
        raw_price = candidate.get("price")
        if raw_price is None:
            continue
        price = _parse_advert_amount(str(raw_price))
        if not _advert_price_in_bounds(price):
            continue
        raw_currency = candidate.get("priceCurrency")
        currency_code = raw_currency.strip().upper() if isinstance(raw_currency, str) and raw_currency.strip() else None
        return price, currency_code
    return None, None


def _text_fallback_price(text: str) -> tuple[Optional[float], Optional[str]]:
    """Conservative raw-text price scan (see _TEXT_PRICE_PATTERNS): only
    currency-marked amounts count, out-of-bounds matches are skipped, and
    when several patterns match, the one nearest the top of the document
    wins (listings put the asking price in the header; footers are full of
    unrelated amounts). Returns (price, currency) or (None, None).
    """
    best: Optional[tuple[int, float, str]] = None  # (position, price, currency)
    for pattern, fixed_currency in _TEXT_PRICE_PATTERNS:
        for match in pattern.finditer(text):
            price = _parse_advert_amount(match.group(1))
            if not _advert_price_in_bounds(price):
                continue  # keep scanning: a later match of this pattern may be sane
            currency_code = fixed_currency or match.group(2).upper()
            if best is None or match.start() < best[0]:
                best = (match.start(), price, currency_code)
            break  # first in-bounds match per pattern; position decides across patterns
    if best is None:
        return None, None
    return best[1], best[2]


def extract_advert_facts_from_html(
    html_content: bytes | str, parser: Optional[_AdvertPageParser] = None
) -> dict:
    """Extract the advert's OWN stated year and asking price from a
    listing page's raw HTML, for the predicted-vs-advertised comparison
    (see the module docstring's "Advert fact extraction" section for the
    full priority order and the "never guess" rule).

    Returns {"year": int|None, "price": float|None, "currency": str|None};
    each field independently stays None when nothing confident was found.
    Pass an already-built `parser` (from _parse_advert_page) to skip
    re-parsing the same page. Pure function (no I/O), directly
    unit-testable -- see tests/test_advert_facts.py. Never raises:
    malformed pages degrade to all-None.
    """
    facts: dict = {"year": None, "price": None, "currency": None}
    if parser is None:
        parser = _parse_advert_page(html_content)
    if parser is None:
        return facts

    # 1. JSON-LD Car/Vehicle/Product nodes -- the most structured source.
    for node in parser.ldjson_nodes:
        if not _jsonld_node_is_advert_type(node):
            continue
        if facts["year"] is None:
            facts["year"] = _jsonld_year(node)
        if facts["price"] is None:
            price, currency_code = _jsonld_price(node)
            if price is not None:
                facts["price"], facts["currency"] = price, currency_code
        if facts["year"] is not None and facts["price"] is not None:
            return facts

    # 2. Meta/OpenGraph fallbacks.
    if facts["year"] is None:
        for title in (parser.og_title, parser.title):
            if title:
                facts["year"] = _last_year_in_text(title)
                if facts["year"] is not None:
                    break
    if facts["price"] is None and parser.meta_price_amount:
        price = _parse_advert_amount(parser.meta_price_amount)
        if _advert_price_in_bounds(price):
            raw_currency = parser.meta_price_currency
            facts["price"] = price
            facts["currency"] = raw_currency.strip().upper() if raw_currency and raw_currency.strip() else None

    # 3. Conservative raw-text fallback -- price only (a bare 4-digit number
    # in page text is far too ambiguous to ever trust as a year).
    if facts["price"] is None:
        price, currency_code = _text_fallback_price(_decode_html(html_content))
        if price is not None:
            facts["price"], facts["currency"] = price, currency_code

    return facts


def _attach_advert_facts(response: PredictionResponse, html_body: bytes, parser: Optional[_AdvertPageParser]) -> None:
    """Populate the advert_* fields of an advert-page PredictionResponse in
    place from the page's own stated facts (extract_advert_facts_from_html)
    plus a GBP conversion (serving/currency.py).

    Never raises, and never leaves the response half-broken: this whole
    comparison is a bonus on top of an already-successful prediction, so
    any failure here (parse, extraction, FX rates) just means the advert_*
    fields stay None and the UI omits the comparison.
    """
    try:
        facts = extract_advert_facts_from_html(html_body, parser=parser)
        if facts.get("year") is not None:
            response.advert_year = int(facts["year"])
        price, currency_code = facts.get("price"), facts.get("currency")
        # A price without a currency can't be honestly compared against a
        # GBP prediction, so it's dropped rather than displayed ambiguously.
        if price is not None and currency_code:
            response.advert_price_original = AdvertPriceOriginal(amount=float(price), currency=currency_code)
            price_gbp = currency.to_gbp(price, currency_code)
            if price_gbp is not None:
                response.advert_price_gbp = round(price_gbp, 2)
                response.rate_source = currency.get_rate_source()
    except Exception:
        logger.exception("Advert fact extraction failed; returning the prediction without the comparison.")


def _run_prediction_for_photos(bundle: ModelBundle, photo_urls: list[str], page_url: str) -> PredictionResponse:
    """Multi-photo path for POST /predict-url when the supplied URL turned
    out to be an advert/listing page rather than a direct image link (see
    extract_photo_urls_from_html). Downloads each candidate photo through
    the *exact same* SSRF-gated `_fetch_image_safely` used for a
    user-supplied direct-image URL -- these URLs came out of
    attacker-controlled page content, so they must not become an SSRF
    bypass -- runs a cheap z-only forward pass on each (no Grad-CAM; see
    _predict_z_only), aggregates with `_aggregate_photo_predictions`
    (median in z-space; see its docstring), and computes Grad-CAM only for
    the resulting representative photo.

    Raises HTTPException(422) if every photo fails to fetch/decode/predict.
    """
    per_photo: list[PerPhotoPrediction] = []
    z_vectors: list[tuple[float, float]] = []
    images_by_index: dict[int, Image.Image] = {}

    for url in photo_urls:
        try:
            image_bytes = _fetch_image_safely(url)  # SAME SSRF gate as a user-supplied URL; see docstring above.
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            logger.info("Skipping unusable advert-page photo: %s", url)
            continue

        try:
            year_z, log_price_z = _predict_z_only(bundle, image)
        except Exception:
            logger.exception("Inference failed for advert-page photo %s; skipping.", url)
            continue

        idx = len(z_vectors)
        z_vectors.append((year_z, log_price_z))
        images_by_index[idx] = image
        real = _destandardize(year_z, log_price_z, bundle.target_norm)
        per_photo.append(
            PerPhotoPrediction(
                url=url,
                year=int(round(real["year"])),
                price_gbp=round(real["price_gbp"] / 10.0) * 10.0,
            )
        )

    if not z_vectors:
        raise HTTPException(status_code=422, detail="Could not find any usable car photos on that page.")

    median_year_z, median_log_price_z, rep_idx = _aggregate_photo_predictions(z_vectors)
    representative_url = per_photo[rep_idx].url

    backend = "onnx" if bundle.onnx_session is not None else "torch"
    note = f"Median of {len(z_vectors)} photo(s) extracted from the advert page."
    gradcam_b64 = None
    if bundle.torch_model is not None:
        try:
            _, gradcam_b64 = _predict_and_gradcam_torch(bundle, images_by_index[rep_idx])
        except Exception:
            logger.exception("Grad-CAM computation failed for the representative advert-page photo.")
        if bundle.onnx_session is not None:
            note += " Prediction served by ONNX; torch backend used only for the representative photo's Grad-CAM."
    else:
        note += " Grad-CAM unavailable: only an ONNX backend is loaded. See TODO(phase 4) in serving/app.py."

    real = _destandardize(median_year_z, median_log_price_z, bundle.target_norm)
    return PredictionResponse(
        year=int(round(real["year"])),
        price_gbp=round(real["price_gbp"] / 10.0) * 10.0,
        gradcam_png_base64=gradcam_b64,
        backend=backend,
        note=note,
        photos_used=len(z_vectors),
        photos_found=len(photo_urls),
        page_url=page_url,
        per_photo=per_photo,
        representative_photo_url=representative_url,
    )


@app.post("/predict-url", response_model=PredictionResponse)
async def predict_url(payload: PredictURLRequest) -> PredictionResponse:
    # SSRF validation runs *before* the model-readiness check: it's cheap,
    # doesn't need the model, and a malformed/unsafe URL should be rejected
    # the same way regardless of whether a model happens to be loaded.
    try:
        body, content_type, final_url = _fetch_bytes_safely(payload.url)
    except URLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = _ensure_model_ready()

    # Content sniffing decides which shape this request is (see the module
    # docstring's "Advert-page photo extraction" section): a direct image
    # keeps the original single-photo behavior; anything that isn't a
    # decodable image but looks like HTML is treated as an advert/listing
    # page and routed through photo extraction + multi-photo aggregation.
    image = _try_decode_image(body)
    if image is not None:
        return _run_prediction(bundle, image)

    if not _looks_like_html(body, content_type):
        raise HTTPException(status_code=400, detail="Downloaded content is not a valid image.")

    # One parse of the page, shared by photo extraction and fact extraction.
    parser = _parse_advert_page(body)
    photo_urls = extract_photo_urls_from_html(body, base_url=final_url, parser=parser)
    if not photo_urls:
        raise HTTPException(status_code=422, detail="Could not find any usable car photos on that page.")

    response = _run_prediction_for_photos(bundle, photo_urls, page_url=payload.url)
    # Bonus comparison against the advert's own stated year/price -- never
    # allowed to fail the prediction itself (see _attach_advert_facts).
    _attach_advert_facts(response, body, parser)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
