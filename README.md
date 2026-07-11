# car-price-vision

**Estimate a car's manufacture year and market price from a single photo — with Grad-CAM interpretability to check *what* the model is actually looking at.**

---

## Problem

Given one photograph of a car (typically an exterior 3-quarter or side shot from a classifieds listing), predict:

1. **Manufacture year** — a regression target, evaluated in mean absolute error (years).
2. **Price** — a regression target on `log(price_gbp)`, evaluated with MAE-log, MAPE, and R².

Both tasks share a single visual backbone and branch into two lightweight regression heads (multi-task learning), on the hypothesis that "what year is this car" and "roughly what is it worth" draw on overlapping visual cues — body style, design era, condition, trim level.

This is framed deliberately as **price estimation**, not "how premium does this car look" — see [Limitations](#limitations) below for why that distinction matters.

## Data

**Primary dataset: [DVM-CAR](https://deepvisualmarketing.github.io/)** (Deep Visual Marketing) — a large-scale UK car marketplace dataset. After joining `Ad_table` with the image index and filtering, the manifest resolves to **1,445,416 usable images** (from ~1.45M raw) across **84 brands**, **858 models**, and **246,175 adverts**, with a year median of **2013** and a price median of **£8,700**. The full manifest is used for EDA and splitting; two checkpoints are reported below on identical leakage-safe splits — an **80,000-image MVP subset** (kept for comparison) and a **300,000-image subset** that scales up the same recipe and is what the live demo currently serves — see [Metrics](#metrics).

> **License note:** DVM-CAR is distributed for **academic, non-commercial research use only**. This repository's MIT license covers the *code* in this repo; it does **not** extend to the dataset, any derived images, or model weights trained on it. Do not use the dataset or derived artifacts commercially. See the dataset's own site for the exact terms.

**Fallback dataset: [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)** — used if DVM-CAR access/staging is delayed or infeasible on the training server. Stanford Cars has make/model/year labels but no price field, so a fallback run would train the year head only (or use crude price proxies), documented at the point it's actually used.

Both datasets are consumed through a single **unified manifest CSV** (see `scripts/build_manifest.py`) with columns:

| column | type | meaning |
|---|---|---|
| `image_path` | str | path to the image file |
| `year` | int | manufacture year |
| `price_gbp` | float | advertised price in GBP |
| `model` | str | car model name |
| `brand` | str | car brand/make |
| `advert_year` | int | year the advert was posted |

**Front-view labeling.** `is_front` originally meant "this file lives in the `images/confirmed_fronts` directory" — but the manifest is built with `--source resized` (scanning `images/resized_DVM`, not `confirmed_fronts`), so no row could ever satisfy that literally and `is_front` was `False` for every one of the 1,445,416 rows. Fixed in `scripts/build_manifest.py`: a `resized_DVM` image is now marked front if its basename also appears in the curated `confirmed_fronts` set (61,248 of 1,445,416 rows match). DVM-CAR's own per-image `viewpoint` field (which nominally encodes camera angle, e.g. 0° = front) is noisy — some 0°-labeled shots are actually rear/boot photos — so `confirmed_fronts` (via `is_front`), not `viewpoint`, is treated as the front-view source of truth for the [logo-mask ablation](#logo-mask-ablation) and the live demo's sample gallery.

## Approach

- **Backbone**: one pretrained ImageNet backbone (default `convnext_tiny`; `efficientnet_v2_s` and `vit_b_16` also supported), classifier head stripped, pooled features shared by both heads.
- **Heads**: two small MLP regression heads — one for `year`, one for `log(price_gbp)`.
- **Loss**: weighted sum of Huber loss on each head (`w_year * L_year + w_price * L_price`), computed on z-standardized targets, robust to outlier listings.
- **Two-stage transfer learning**:
  1. **Stage 1** — freeze the backbone, train only the heads (5 epochs; fast convergence, stable initialization).
  2. **Stage 2** — unfreeze the last few backbone blocks, fine-tune end-to-end at a lower learning rate (15 epochs).
- **Splits**: leakage-safe by construction — see `src/car_price_vision/data/splits.py`, grouped by advert id so no advert's images leak across splits. Two modes: `by_advert` (realistic deployment scenario) and `by_model` (strict unseen-models holdout — 10% of models held out entirely, used specifically to test generalization beyond memorized model/brand priors).
- **Training runs**: `convnext_tiny`, 224×224 input, AdamW optimizer, bf16 automatic mixed precision, on an NVIDIA RTX PRO 6000 Blackwell. Two checkpoints under the identical recipe: an **80,000-image subset** (`configs/default.yaml`, ~8 minutes) and a **300,000-image subset** (`configs/convnext_300k.yaml` — same hyperparameters, only `subset_size` and output paths differ, ~27 minutes). **The 300k checkpoint is what [https://pricevision.app](https://pricevision.app) currently serves.**

## Metrics

Reported on val / test / unseen-models holdout:

| Metric | Head | Meaning |
|---|---|---|
| MAE (years) | year | mean absolute error in manufacture year |
| MAE-log | price | mean absolute error in log-price space |
| MAPE | price | mean absolute % error, computed on `exp(log price)` |
| R² | both | coefficient of determination |
| within-brand price correlation | price | Pearson correlation of predicted vs. true price, computed *within* each brand — see [Limitations](#limitations) |

Both checkpoints are `convnext_tiny`, trained per the [Approach](#approach) above on identical leakage-safe splits — differing only in how much of the manifest they trained on. **The 300k checkpoint is the one deployed at [https://pricevision.app](https://pricevision.app)**; training curves are in `results/convnext_300k/metrics.csv`, full per-brand breakdown in `results/convnext_300k/eval_metrics.json` (80k baseline: `results/convnext/`).

| Subset | Split | MAE-years | MAE-log | MAPE | R² (year) | R² (price) | within-brand corr |
|---|---|---|---|---|---|---|---|
| 80k (baseline) | val | 1.62 | 0.332 | 35.9% | 0.764 | 0.800 | 0.714 |
| 80k (baseline) | test | 1.61 | 0.335 | 35.4% | 0.763 | 0.794 | 0.631 |
| 80k (baseline) | unseen-models holdout | 1.94 | 0.394 | 44.0% | 0.651 | 0.712 | 0.472 |
| 300k (**deployed**) | val | 1.313 | 0.278 | 30.7% | 0.838 | 0.852 | 0.654 |
| 300k (**deployed**) | test | 1.327 | 0.276 | 30.6% | 0.835 | 0.851 | 0.697 |
| 300k (**deployed**) | unseen-models holdout | 1.727 | 0.375 | 44.3% | 0.736 | 0.762 | 0.449 |

**Scaling 80k → 300k** cut test MAPE from 35.4% to 30.6% and lifted test R²(price) from 0.794 to 0.851 — more data clearly helps in-distribution. But unseen-models holdout MAPE barely moved (44.0% → 44.3%) and holdout within-brand correlation actually dropped slightly (0.472 → 0.449). Generalizing to car models the network has never seen images of remains the hard part of this task, and 4× more training images did not fix it.

## Results

**Backbone comparison (Phase 3, 80k subset).** `efficientnet_v2_s`, trained under the identical data, splits, and two-stage schedule, scored marginally better on point metrics (test MAE-years 1.54, MAPE 34.9%, R² price 0.800, within-brand corr 0.662) — but the gap is within run-to-run noise. `convnext_tiny` was chosen as the deployed backbone for comparable accuracy with a cleaner convolutional Grad-CAM target and lower CPU-inference latency; it's the one later scaled up to the 300k subset (see [Metrics](#metrics)).

**Interpretability finding (Phase 4).** Grad-CAM for the price head concentrates on the wheels/alloys and body proportions rather than the badge/grille, and the within-brand predicted-vs-true price correlation stays clearly positive on the deployed 300k checkpoint (0.697 on test, 0.449 on unseen-models holdout). Together this is evidence — not proof — that the model is picking up genuine within-brand price signal rather than relying purely on a brand-lookup shortcut; the [logo-mask ablation](#logo-mask-ablation) below tests this more directly, and [Limitations](#limitations) covers what neither fully rules out.

**Sample live predictions** (through the deployed API):

| Car | Actual | Predicted |
|---|---|---|
| Peugeot 108 (2017) | 2017, £8,700 | 2017, £8,150 |
| Porsche Macan (2018) | 2018, £53,000 | 2018, £32,060 |
| Ford Fiesta (2005) | 2005, £1,495 | 2003, £830 |

*(Captured against the 80k-subset checkpoint, before the 300k model was deployed; illustrative of the pattern below rather than exact current-model output.)* The Porsche Macan case is illustrative of a systematic pattern (see [Limitations](#limitations)): the model correctly recognizes an "expensive-looking" car and gets the year right, but regresses the price toward the mean, understating high-price-tail listings.

## Interpretability

`src/car_price_vision/interpret/gradcam.py` implements **Grad-CAM** from scratch (forward/backward hooks on a target conv layer — no dependency on the external `grad-cam` package), targeting the last stage of the CNN backbone by default. It's used to:

- Visualize which pixels drive the `price` prediction for individual cars (`notebooks/04_interpretability.ipynb`).
- Check whether attention concentrates on the badge/grille (a shortcut) vs. spreads across body panels and proportions (genuine visual reasoning). In practice, on the deployed ConvNeXt-Tiny model, attention concentrates on wheels/alloys and body proportions — see [Results](#results).
- Feed a causal ablation (`scripts/ablation_logo_mask.py`) that masks the badge/grille region — and, as a control, a same-area background box — and measures the resulting prediction shift and accuracy delta. See [Logo-Mask Ablation](#logo-mask-ablation) below.

## Logo-Mask Ablation

Grad-CAM shows *where* attention concentrates, but that's correlational. `scripts/ablation_logo_mask.py` runs a controlled ablation instead: on confirmed front-view photos (`is_front=True` — see [Data](#data) for why that's the trustworthy front-view flag), the same **300k deployed checkpoint** is evaluated three times per image, with no change to the model or its weights:

- **none** — unmodified image (baseline).
- **badge** — a fixed box over the badge/grille filled with the ImageNet mean color.
- **control** — a same-area box in the top-left corner (background, not the car), filled the same way — the causal control, since it removes the same *amount* of pixel information from a location that carries no brand signal.

This is a diagnostic on a frozen checkpoint, not a retraining exercise — the deployed model always sees the full, unmasked image. The question it answers is narrower than "does the model use the badge" (it should, and legitimately can — brand positioning is real price information): it's whether the model has degenerated into a badge→price *lookup* that ignores everything else.

| Split (n) | Condition | MAE-years | MAPE | R² (price, log) |
|---|---|---|---|---|
| test (1,094) | none | 1.044 | 21.1% | 0.889 |
| test (1,094) | badge | 1.338 | 29.6% | 0.807 |
| test (1,094) | control | 1.064 | 21.9% | 0.887 |
| holdout (1,284) | none | 1.567 | 36.6% | 0.721 |
| holdout (1,284) | badge | 1.720 | 39.4% | 0.677 |
| holdout (1,284) | control | 1.586 | 37.6% | 0.716 |

Full numbers: `results/ablation_300k/summary.md`, `results/ablation_300k/ablation_test.json`, `results/ablation_300k/ablation_holdout.json`.

**Reading it.** On test, masking the badge costs +0.29 MAE-years and +8.5pp MAPE versus the unmasked baseline, and shifts the predicted price by more than 10% on ~68% of images; masking the same-sized control box costs +0.02 MAE-years and +0.8pp MAPE — an order of magnitude less. The same asymmetry holds on the (harder) holdout split, just smaller in absolute terms: badge +0.15 yr / +2.8pp MAPE vs. control +0.02 yr / +1.0pp MAPE.

![Logo-mask ablation example: none vs. badge-masked vs. control-masked](results/ablation_300k/examples/example_00.png)

**Interpretation.** Badge-masking clearly hurts more than control-masking, at roughly the same masked area — that answers "no, it's not a pure badge lookup": the badge region carries a real but *minority* share of the price signal, and the model still works reasonably without it. Even with the badge gone, badge-masked accuracy stays in the same broad range as the deployed model's normal unmasked performance (test: 29.6% MAPE / R² 0.807 masked vs. 30.6% MAPE / R² 0.851 unmasked on the full test set; holdout: 39.4% MAPE / R² 0.677 masked vs. 44.3% MAPE / R² 0.762 unmasked) — and the within-brand correlation reported in [Metrics](#metrics) (0.697 test) confirms the model still ranks prices *within* a brand, precisely where the badge can't help it.

**Future work.** A brand-invariant "design-only" model — via logo detection plus train-time masking across all viewpoints, not just confirmed fronts — would let this ablation become a training-time constraint rather than a post-hoc diagnostic.

## Limitations

Documented honestly, up front, rather than discovered at the end:

1. **Brand/logo shortcut.** The model may learn to read the badge or grille shape and effectively predict price via "which brand is this" rather than genuine design/condition cues. Mitigations built into this repo: Grad-CAM on the badge region, within-brand price correlation as a diagnostic (if it collapses toward zero while overall correlation looks strong, that's a shortcut signal), and the [logo-mask ablation](#logo-mask-ablation), which occludes the badge/grille and compares the prediction shift against a same-area control box. The ablation confirms the badge carries a real but minority share of the price signal — masking it costs +8.5pp test MAPE vs. +0.8pp for masking an equally-sized control region — and the observed within-brand correlations (0.697 test / 0.449 holdout on the deployed 300k checkpoint, both clearly above zero) further *mitigate* this concern but do not fully eliminate it — a real, if weaker, shortcut component could still be present underneath a genuine signal.
2. **Price ≠ perceived premium-ness.** The price target is a market-observed advertised price (UK classifieds), which reflects supply/demand, mileage, condition, and seller behavior — not a subjective "how upscale does this car look" judgment. This is framed strictly as *price estimation*, not aesthetic or brand-prestige scoring.
3. **UK-market bias.** DVM-CAR reflects UK second-hand car market pricing and vehicle mix; predictions should not be assumed to transfer to other markets (LHD/RHD, different tax regimes, different popular models).
4. **GBP price inflation over time.** Advertised prices span multiple years, and nominal GBP prices drift upward over time independent of the car itself. `advert_year` is carried through the manifest specifically to allow controlling for this (e.g. as a covariate or via inflation-adjusted price normalization) — full mitigation is a phase 2+ modeling decision, not yet implemented in the MVP loss.
5. **Photo-only age→price leakage.** A car's visible age (paint condition, wear, wheel style, headlight design era) is itself informative about both `year` and `price` in ways that are legitimate for this task (that's the whole premise) but also means the two heads' errors are correlated — a systematic year-prediction bias will likely show up as a price bias too. Residual analysis by car age is included in the interpretability notebook.
6. **Generalization gap to unseen models.** On the deployed 300k checkpoint, MAPE degrades from ~30.6% on val/test to 44.3% on the unseen-models holdout (R² price 0.851 → 0.762), and within-brand correlation drops from 0.697 to 0.449. Scaling training data 80k → 300k cut in-distribution error substantially (test MAPE 35.4% → 30.6%) but barely moved this holdout gap (holdout MAPE was already 44.0% at 80k) — the model is meaningfully weaker on car models it has never seen images of, and more training data alone does not close that gap. This should temper confidence in predictions for rare or unusual models.
7. **High-price-tail underestimation.** The model tends to regress high-value cars toward the mean rather than tracking the full price range — e.g. a Porsche Macan actually listed at £53,000 was predicted at £32,060 (see [Results](#results)). Predictions for premium/luxury vehicles should be treated as directionally correct at best, not as reliable point estimates.

## Project Structure

```
car-price-vision/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml           # 80k MVP baseline
│   └── convnext_300k.yaml     # 300k scale-up, deployed checkpoint
├── src/car_price_vision/
│   ├── data/                # dataset, leakage-safe splits, transforms
│   ├── models/               # backbone, regression heads, multi-task model
│   ├── interpret/             # Grad-CAM
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py               # two-stage training loop
│   ├── eval.py                # checkpoint evaluation
│   └── utils.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_finetune.ipynb
│   └── 04_interpretability.ipynb
├── scripts/
│   ├── inventory_rtx.sh       # phase 0: training server inventory
│   ├── build_manifest.py      # raw DVM-CAR metadata -> unified manifest CSV
│   └── ablation_logo_mask.py  # logo-mask ablation, see Logo-Mask Ablation
├── results/                    # eval_metrics.json / metrics.csv per run, ablation_300k/
├── tests/                      # 71 pytest tests, see Tests
└── serving/
    ├── app.py                 # FastAPI: /predict, /predict-url, /gallery, /health
    ├── static/index.html
    ├── static/gallery/         # 144 sample photos + gallery.json
    ├── Dockerfile
    └── requirements.txt
```

## Setup / Usage

```bash
# Clone and install
cd car-price-vision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Phase 0: inventory the training server (run on `rtx`)
bash scripts/inventory_rtx.sh

# Phase 1: build the manifest from raw DVM-CAR metadata (fill in TODOs first)
python scripts/build_manifest.py --raw-dir /path/to/dvm-car/raw --data-root /path/to/dvm-car/images --out data/manifest.csv

# Train (two-stage: frozen backbone -> fine-tune)
python -m car_price_vision.train --config configs/default.yaml

# Evaluate a checkpoint
python -m car_price_vision.eval --config configs/default.yaml \
    --checkpoint checkpoints/default_run/latest.pt --splits val test holdout

# Run the demo API locally
uvicorn serving.app:app --reload --app-dir serving
```

**Live demo:** **[https://pricevision.app](https://pricevision.app)** — three ways to get a prediction:

- **Upload a photo** (drag-and-drop or paste) — get predicted year, price, and a Grad-CAM overlay.
- **Pick from the gallery** — 144 sample photos, all front-view shots of cars from the **unseen-models holdout split** (car models the network never saw during training). After a prediction, the UI reveals the advert's actual year/price next to it, so you can see how far off the model is on a genuinely unseen model.
- **Predict from a URL** — paste an image URL and the server fetches it itself.

Endpoints (see `serving/app.py`'s module docstring for the exact contract):

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | serves `static/index.html` |
| `/health` | GET | model-loaded status + `model_info` (backbone, img_size, and the `metrics` block from `model_meta.json`, when present) |
| `/gallery` | GET | sample-photo metadata: file, brand, model, true year/price, bodytype, color |
| `/predict` | POST | multipart image upload → JSON prediction + base64 Grad-CAM PNG |
| `/predict-url` | POST | `{"url": "..."}` → same prediction, image fetched server-side |

`/predict-url` is a classic SSRF surface (a visitor asks *this server* to fetch an arbitrary URL on its behalf), so it's locked down: only `http`/`https` schemes; every resolved DNS record (not just the first) is checked and rejected if it's a private, loopback, or link-local address; redirects are only followed to the same host; and the fetch is capped at 10 MB / 10 s with content-type sniffing. Test coverage: `tests/test_serving_url_safety.py` (see [Tests](#tests)).

Served via FastAPI behind a Cloudflare Tunnel (no open inbound ports); CPU inference (ONNX for the point prediction, PyTorch for the Grad-CAM overlay). Deployed artifacts: `models/model.onnx`, `models/model.pt`, `models/model_meta.json` — the latter now carries a `metrics` block that `/health` exposes and the UI displays alongside each prediction. Gallery images live in `serving/static/gallery/` (~2.5 MB), under the same non-commercial restriction as the underlying dataset — see [License / Dataset Note](#license--dataset-note).

## Tests

`tests/` has 71 pytest tests covering the things most likely to silently break: leakage invariants of `make_splits` (both `by_advert` and the unseen-model `by_model` holdout mode), metric correctness against hand-computed reference values, target z-normalization round-tripping, transform output shapes, and SSRF URL-validation for the `/predict-url` serving endpoint.

```bash
pip install -r requirements.txt   # includes pytest
python3 -m pytest tests/ -q
```

## Roadmap

- **Phase 0 — Infra & inventory.** Confirm `rtx` GPU/disk/docker/python environment (`scripts/inventory_rtx.sh`).
- **Phase 1 — Data.** Stage DVM-CAR, build the unified manifest (`scripts/build_manifest.py`), EDA (`notebooks/01_eda.ipynb`), finalize leakage-safe splits.
- **Phase 2 — Baseline.** Frozen-backbone training (Stage 1), first metrics, backbone comparison (`notebooks/02_baseline.ipynb`, `03_finetune.ipynb`).
- **Phase 3 — Fine-tuning & interpretability.** Full two-stage training, Grad-CAM gallery, brand-shortcut diagnostics, unseen-models holdout evaluation (`notebooks/04_interpretability.ipynb`).
- **Phase 4 — Export & serving.** Export best checkpoint to ONNX, wire up `serving/app.py`, containerize.
- **Phase 5 — Deploy.** ✅ Done. FastAPI+ONNX-CPU demo shipped to a VPS, published via a Cloudflare Tunnel (no open inbound ports) — live at [https://pricevision.app](https://pricevision.app), see `serving/deploy/`. Since then: scaled the deployed checkpoint from 80k to 300k training images, added gallery/upload/predict-URL modes to the demo UI (see [Setup / Usage](#setup--usage)), and ran a real logo-mask ablation (see [Logo-Mask Ablation](#logo-mask-ablation)) in place of the earlier "optional ablation" placeholder.

## License / Dataset Note

Code in this repository is licensed under [MIT](LICENSE). The DVM-CAR dataset (and any derived images, features, or trained weights) is subject to its own **academic, non-commercial** license and is **not** covered by the MIT license — see the [Data](#data) section above.
