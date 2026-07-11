"""Currency conversion for advert-page price comparisons.

Feature: when POST /predict-url resolves an advert/listing page (see
`extract_advert_facts_from_html` in serving/app.py), the page usually quotes
its own price in whatever currency the marketplace uses (USD, EUR, UAH, ...).
To show that figure next to this model's GBP-denominated prediction, it has
to be converted to GBP first -- that's all this module does.

Rate source
------------
Live rates come from `RATES_URL` (https://open.er-api.com/v6/latest/GBP), a
free, no-API-key FX-rate endpoint that returns `{"rates": {"USD": 1.27, ...}}`
i.e. units of each currency per 1 GBP. This host is fixed by us and baked
into this module -- it is NOT a user-supplied URL -- so it deliberately does
NOT go through app.py's SSRF gate (`validate_image_url`/`_fetch_bytes_safely`
exist specifically because /predict-url's URL comes from an untrusted
visitor; this one never varies).

Rates are fetched lazily (only on first use, not at import time) and cached
in-memory for CACHE_TTL_SECONDS; a module-level lock keeps the read-check-
fetch-write sequence safe enough for FastAPI's default sync-endpoint
threadpool (this module doesn't need anything fancier than that for a
demo project).

Failure handling
------------------
A currency lookup is a "nice to have" for one comparison row in the UI --
it must never be allowed to break an otherwise-successful prediction. Every
public function here is wrapped so any failure (network error, unexpected
response shape, unknown currency code) returns None instead of raising;
callers in app.py should treat None as "omit this part of the response",
never as an error worth surfacing to the visitor.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger("car_price_vision.serving.currency")

RATES_URL = "https://open.er-api.com/v6/latest/GBP"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h
FETCH_TIMEOUT_SECONDS = 5.0

# Approximate snapshot (units of currency per 1 GBP), used only when the
# live API is unreachable *and* there is no cached response yet to fall
# back on (e.g. offline dev, CI, first request after a network blip, or
# tests -- see tests/test_advert_facts.py which monkeypatches `_cache`
# directly with these instead of touching the network at all). Good enough
# for a "roughly how much is this" comparison row; not a financial feed.
FALLBACK_RATES = {"USD": 1.27, "EUR": 1.17, "UAH": 53.0, "GBP": 1.0}

_lock = threading.Lock()
# {"rates": {...}, "fetched_at": float (time.monotonic()), "source": "live"|"fallback"}
_cache: Optional[dict] = None


def _fetch_live_rates() -> Optional[dict]:
    """Best-effort GET of RATES_URL. Returns the "rates" dict on success,
    None on any error (network, non-2xx, unexpected JSON shape) -- never
    raises.
    """
    try:
        response = httpx.get(RATES_URL, timeout=FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        rates = data.get("rates")
        if not isinstance(rates, dict) or "USD" not in rates:
            logger.warning("Unexpected response shape from %s; ignoring.", RATES_URL)
            return None
        return rates
    except Exception:
        logger.warning("Could not fetch live FX rates from %s; falling back.", RATES_URL, exc_info=True)
        return None


def _get_rates() -> tuple[dict, str]:
    """Return (rates_dict, source) where source is "live" or "fallback",
    refreshing from the network if the cache is empty or older than
    CACHE_TTL_SECONDS. Thread-safe via `_lock` (see module docstring).
    """
    global _cache
    with _lock:
        now = time.monotonic()
        if _cache is not None and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
            return _cache["rates"], _cache["source"]

        live_rates = _fetch_live_rates()
        if live_rates is not None:
            _cache = {"rates": live_rates, "fetched_at": now, "source": "live"}
            return live_rates, "live"

        if _cache is not None:
            # Stale cache beats no data: keep serving what we had rather
            # than silently switching to the hardcoded snapshot just
            # because one refresh attempt failed.
            return _cache["rates"], _cache["source"]

        _cache = {"rates": FALLBACK_RATES, "fetched_at": now, "source": "fallback"}
        return FALLBACK_RATES, "fallback"


def get_rate_source() -> str:
    """"live" or "fallback" -- whichever `_get_rates` is currently serving.
    Used by app.py to populate PredictionResponse.rate_source alongside a
    to_gbp() conversion.
    """
    try:
        _, source = _get_rates()
        return source
    except Exception:
        logger.exception("Failed to determine FX rate source.")
        return "fallback"


def to_gbp(amount: Optional[float], currency: Optional[str]) -> Optional[float]:
    """Convert `amount` in `currency` (ISO 4217 code, e.g. "USD") to GBP.

    Returns None for any missing/invalid input, an unrecognized currency
    code, or if rates can't be obtained at all -- see module docstring for
    why this never raises.
    """
    if amount is None or not currency:
        return None
    try:
        rates, _source = _get_rates()
        code = currency.strip().upper()
        if code == "GBP":
            return float(amount)
        rate = rates.get(code)
        if not rate:
            return None
        # `rates[code]` is "units of `code` per 1 GBP" (GBP is the API's
        # base currency), so GBP = amount / rate.
        return float(amount) / float(rate)
    except Exception:
        logger.exception("Currency conversion failed for amount=%r currency=%r", amount, currency)
        return None
