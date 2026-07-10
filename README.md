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

**Primary dataset: [DVM-CAR](https://deepvisualmarketing.github.io/)** (Deep Visual Marketing) — a large-scale UK car marketplace dataset with ~1.45M images across ~900 car models, paired with structured metadata (price, year, model, brand, advert date, and more).

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

## Approach

- **Backbone**: one pretrained ImageNet backbone (default `convnext_tiny`; `efficientnet_v2_s` and `vit_b_16` also supported), classifier head stripped, pooled features shared by both heads.
- **Heads**: two small MLP regression heads — one for `year`, one for `log(price_gbp)`.
- **Loss**: weighted sum of Huber loss on each head (`w_year * L_year + w_price * L_price`), robust to outlier listings.
- **Two-stage transfer learning**:
  1. **Stage 1** — freeze the backbone, train only the heads (fast convergence, stable initialization).
  2. **Stage 2** — unfreeze the last few backbone blocks, fine-tune end-to-end at a lower learning rate.
- **Splits**: leakage-safe by construction — see `src/car_price_vision/data/splits.py`. Two modes: `by_advert` (realistic deployment scenario) and `by_model` (strict unseen-models holdout, used specifically to test generalization beyond memorized model/brand priors).

## Metrics

Reported on val / test / unseen-models holdout:

| Metric | Head | Meaning |
|---|---|---|
| MAE (years) | year | mean absolute error in manufacture year |
| MAE-log | price | mean absolute error in log-price space |
| MAPE | price | mean absolute % error, computed on `exp(log price)` |
| R² | both | coefficient of determination |
| within-brand price correlation | price | Pearson correlation of predicted vs. true price, computed *within* each brand — see [Limitations](#limitations) |

Result tables are intentionally left as `TBD` in this repository until real training runs are complete — no fabricated numbers.

| Split | MAE-years | MAE-log | MAPE | R² (year) | R² (price) | within-brand corr |
|---|---|---|---|---|---|---|
| val | TBD | TBD | TBD | TBD | TBD | TBD |
| test | TBD | TBD | TBD | TBD | TBD | TBD |
| unseen-models holdout | TBD | TBD | TBD | TBD | TBD | TBD |

## Interpretability

`src/car_price_vision/interpret/gradcam.py` implements **Grad-CAM** from scratch (forward/backward hooks on a target conv layer — no dependency on the external `grad-cam` package), targeting the last stage of the CNN backbone by default. It's used to:

- Visualize which pixels drive the `price` prediction for individual cars (`notebooks/04_interpretability.ipynb`).
- Check whether attention concentrates on the badge/grille (a shortcut) vs. spreads across body panels and proportions (genuine visual reasoning).
- Support an optional logo-mask ablation: mask the badge/grille region and measure the prediction shift.

## Limitations

Documented honestly, up front, rather than discovered at the end:

1. **Brand/logo shortcut.** The model may learn to read the badge or grille shape and effectively predict price via "which brand is this" rather than genuine design/condition cues. Mitigations built into this repo: Grad-CAM on the badge region, within-brand price correlation as a diagnostic (if it collapses toward zero while overall correlation looks strong, that's a shortcut signal), and an optional logo-mask ablation.
2. **Price ≠ perceived premium-ness.** The price target is a market-observed advertised price (UK classifieds), which reflects supply/demand, mileage, condition, and seller behavior — not a subjective "how upscale does this car look" judgment. This is framed strictly as *price estimation*, not aesthetic or brand-prestige scoring.
3. **UK-market bias.** DVM-CAR reflects UK second-hand car market pricing and vehicle mix; predictions should not be assumed to transfer to other markets (LHD/RHD, different tax regimes, different popular models).
4. **GBP price inflation over time.** Advertised prices span multiple years, and nominal GBP prices drift upward over time independent of the car itself. `advert_year` is carried through the manifest specifically to allow controlling for this (e.g. as a covariate or via inflation-adjusted price normalization) — full mitigation is a phase 2+ modeling decision, not yet implemented in the MVP loss.
5. **Photo-only age→price leakage.** A car's visible age (paint condition, wear, wheel style, headlight design era) is itself informative about both `year` and `price` in ways that are legitimate for this task (that's the whole premise) but also means the two heads' errors are correlated — a systematic year-prediction bias will likely show up as a price bias too. Residual analysis by car age is included in the interpretability notebook.

## Project Structure

```
car-price-vision/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
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
│   └── build_manifest.py      # raw DVM-CAR metadata -> unified manifest CSV
└── serving/
    ├── app.py                 # FastAPI: POST /predict -> year, price, Grad-CAM
    ├── static/index.html
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

**Live demo:** [https://aetherkin.space](https://aetherkin.space) (coming soon — FastAPI on VPS, currently a placeholder page while the model is in training).

## Roadmap

- **Phase 0 — Infra & inventory.** Confirm `rtx` GPU/disk/docker/python environment (`scripts/inventory_rtx.sh`).
- **Phase 1 — Data.** Stage DVM-CAR, build the unified manifest (`scripts/build_manifest.py`), EDA (`notebooks/01_eda.ipynb`), finalize leakage-safe splits.
- **Phase 2 — Baseline.** Frozen-backbone training (Stage 1), first metrics, backbone comparison (`notebooks/02_baseline.ipynb`, `03_finetune.ipynb`).
- **Phase 3 — Fine-tuning & interpretability.** Full two-stage training, Grad-CAM gallery, brand-shortcut diagnostics, unseen-models holdout evaluation (`notebooks/04_interpretability.ipynb`).
- **Phase 4 — Export & serving.** Export best checkpoint to ONNX, wire up `serving/app.py`, containerize.
- **Phase 5 — Deploy.** Ship the FastAPI+ONNX-CPU demo to a VPS behind the existing Traefik reverse proxy.

## License / Dataset Note

Code in this repository is licensed under [MIT](LICENSE). The DVM-CAR dataset (and any derived images, features, or trained weights) is subject to its own **academic, non-commercial** license and is **not** covered by the MIT license — see the [Data](#data) section above.
