"""Tests for the advert-page "own figures" extraction added to POST
/predict-url (serving/app.py: `extract_advert_facts_from_html`) and its GBP
conversion helper (serving/currency.py: `to_gbp`).

These are pure unit tests: no real network access anywhere --
`extract_advert_facts_from_html` only parses inline HTML fixtures, and the
currency tests monkeypatch serving/currency's in-memory rate cache (with the
module's own hardcoded fallback snapshot) so `to_gbp` never fetches live
rates. See tests/test_serving_page_extract.py for the photo-extraction tests
these sit alongside (same one-parse-per-page machinery).
"""

from __future__ import annotations

import time

import pytest

from serving import currency
from serving.app import extract_advert_facts_from_html

# --- extract_advert_facts_from_html: JSON-LD ----------------------------


def test_jsonld_car_with_offers_price_currency_and_year():
    html = """
    <html><head>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Car",
        "name": "Ford Focus",
        "vehicleModelDate": "2016",
        "offers": {"@type": "Offer", "price": "8500", "priceCurrency": "USD"}
      }
      </script>
    </head><body></body></html>
    """
    facts = extract_advert_facts_from_html(html)
    assert facts == {"year": 2016, "price": 8500.0, "currency": "USD"}


def test_jsonld_offers_as_list_and_numeric_price():
    html = """
    <html><head>
      <script type="application/ld+json">
      {
        "@type": "Vehicle",
        "productionDate": "2012-06-01",
        "offers": [{"price": 12000, "priceCurrency": "EUR"}]
      }
      </script>
    </head></html>
    """
    facts = extract_advert_facts_from_html(html)
    assert facts == {"year": 2012, "price": 12000.0, "currency": "EUR"}


def test_jsonld_top_level_price_and_year_from_name():
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product", "name": "Nissan 350Z 2008", "price": "9999", "priceCurrency": "GBP"}
      </script>
    </head></html>
    """
    facts = extract_advert_facts_from_html(html)
    assert facts == {"year": 2008, "price": 9999.0, "currency": "GBP"}


def test_jsonld_non_advert_types_are_ignored():
    # An Organization's founding year / a WebSite's fields must never be
    # mistaken for the car's -- only Car/Vehicle/Product nodes count.
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Organization", "name": "Dealer Since 1985", "price": "777"}
      </script>
    </head></html>
    """
    facts = extract_advert_facts_from_html(html)
    assert facts == {"year": None, "price": None, "currency": None}


# --- extract_advert_facts_from_html: meta / title ------------------------


def test_og_title_year_takes_last_match():
    # "350Z" isn't a valid year-shaped number, and when a title contains two
    # year-shaped numbers the LAST one is the model year by convention.
    html = """
    <html><head>
      <meta property="og:title" content="Nissan 350Z 2008">
    </head></html>
    """
    assert extract_advert_facts_from_html(html)["year"] == 2008

    html_two_years = """
    <html><head>
      <meta property="og:title" content="2008 registered, facelift 2011 - Audi A4">
    </head></html>
    """
    assert extract_advert_facts_from_html(html_two_years)["year"] == 2011


def test_title_tag_year_fallback_when_no_og_title():
    html = "<html><head><title>Skoda Octavia 2019 for sale</title></head><body></body></html>"
    assert extract_advert_facts_from_html(html)["year"] == 2019


def test_year_out_of_range_in_title_is_rejected():
    # 1932 predates the 1950 floor; 2077 is past the 2029 ceiling.
    html = "<html><head><title>Ford Model B 1932, restored for 2077</title></head></html>"
    assert extract_advert_facts_from_html(html)["year"] is None


def test_meta_product_price_amount_and_currency():
    html = """
    <html><head>
      <meta property="product:price:amount" content="12500.00">
      <meta property="product:price:currency" content="EUR">
    </head></html>
    """
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 12500.0
    assert facts["currency"] == "EUR"


# --- extract_advert_facts_from_html: raw-text price fallback -------------


def test_text_fallback_dollar_with_space_separator():
    html = "<html><body><p>Price: $8 500 negotiable</p></body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 8500.0
    assert facts["currency"] == "USD"


def test_text_fallback_hryvnia_with_space_separators():
    html = "<html><body><span>1 250 000 грн</span></body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 1250000.0
    assert facts["currency"] == "UAH"


def test_text_fallback_euro_with_dot_thousand_separator():
    # European formatting: "12.500" is twelve and a half thousand, not 12.5.
    html = "<html><body>Preis: €12.500</body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 12500.0
    assert facts["currency"] == "EUR"


def test_text_fallback_currency_code_suffix():
    html = "<html><body>Asking 7 200 USD or best offer</body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 7200.0
    assert facts["currency"] == "USD"


def test_text_fallback_prefers_match_nearest_top_of_document():
    html = "<html><body><h1>€5 000</h1> ... much later the footer says $8 500</body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 5000.0
    assert facts["currency"] == "EUR"


def test_text_fallback_out_of_bounds_prices_are_rejected():
    # Below the 50 floor and above the 5,000,000 ceiling -- both dropped.
    too_small = "<html><body>only 10 USD</body></html>"
    too_large = "<html><body>$9 999 999 supercar</body></html>"
    assert extract_advert_facts_from_html(too_small)["price"] is None
    assert extract_advert_facts_from_html(too_large)["price"] is None


def test_bare_numbers_without_currency_marker_are_never_a_price():
    html = "<html><body>Mileage 89 000, engine 1 998 cc</body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] is None
    assert facts["currency"] is None


def test_nothing_found_returns_all_none():
    html = "<html><head><title>A car</title></head><body><p>No facts here.</p></body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts == {"year": None, "price": None, "currency": None}


def test_jsonld_beats_title_and_text_fallback():
    # All three tiers present and disagreeing: JSON-LD must win both fields.
    html = """
    <html><head>
      <title>Ford Focus 2010 - $9 999</title>
      <script type="application/ld+json">
      {"@type": "Car", "vehicleModelDate": "2016",
       "offers": {"price": "8500", "priceCurrency": "USD"}}
      </script>
    </head><body>Footer price €1 111</body></html>
    """
    facts = extract_advert_facts_from_html(html)
    assert facts == {"year": 2016, "price": 8500.0, "currency": "USD"}


# --- serving/currency.py: to_gbp -----------------------------------------


@pytest.fixture
def fallback_rates(monkeypatch):
    """Prime serving/currency's in-memory cache with the module's own
    hardcoded fallback snapshot and forbid any live fetch, so these tests
    are deterministic and fully offline.
    """

    def _no_network():  # pragma: no cover - would only fire on a bug
        raise AssertionError("to_gbp must not hit the network when the cache is primed")

    monkeypatch.setattr(currency, "_fetch_live_rates", _no_network)
    monkeypatch.setattr(
        currency,
        "_cache",
        {"rates": dict(currency.FALLBACK_RATES), "fetched_at": time.monotonic(), "source": "fallback"},
    )
    return currency.FALLBACK_RATES


def test_to_gbp_usd_uses_fallback_rate(fallback_rates):
    # 1.27 USD per GBP -> $127 is exactly £100.
    assert currency.to_gbp(127.0, "USD") == pytest.approx(100.0)


def test_to_gbp_uah_and_eur(fallback_rates):
    assert currency.to_gbp(53.0, "UAH") == pytest.approx(1.0)
    assert currency.to_gbp(1170.0, "EUR") == pytest.approx(1000.0)


def test_to_gbp_gbp_is_identity(fallback_rates):
    assert currency.to_gbp(250.0, "GBP") == pytest.approx(250.0)


def test_to_gbp_is_case_insensitive(fallback_rates):
    assert currency.to_gbp(127.0, "usd") == pytest.approx(100.0)


def test_to_gbp_unknown_currency_returns_none(fallback_rates):
    assert currency.to_gbp(100.0, "XYZ") is None


def test_to_gbp_missing_inputs_return_none(fallback_rates):
    assert currency.to_gbp(None, "USD") is None
    assert currency.to_gbp(100.0, None) is None
    assert currency.to_gbp(100.0, "") is None


def test_rate_source_reports_fallback_for_primed_cache(fallback_rates):
    assert currency.get_rate_source() == "fallback"


# ---- marketplace formats added after live-testing against auto.ria ----


def test_text_fallback_suffix_dollar_nbsp():
    """Ukrainian/European rendering puts the symbol AFTER the amount with
    NBSP thousand separators: '10\xa0000 $'."""
    html = "<html><body>ціна 10\xa0000 $ за авто</body></html>"
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 10000.0
    assert facts["currency"] == "USD"


def test_text_fallback_embedded_state_price_keys():
    """Marketplaces inline the asking price in JS state: '"priceUSD":10000'
    (also the JSON-escaped variant 'priceUAH\\":445900')."""
    html = '<html><body><script>window.state={"priceUSD":10000,"priceUAH":445900}</script></body></html>'
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 10000.0
    assert facts["currency"] == "USD"

    html_escaped = "<html><body><script>x=\"{\\\"priceUAH\\\":445900}\"</script></body></html>"
    facts = extract_advert_facts_from_html(html_escaped)
    assert facts["price"] == 445900.0
    assert facts["currency"] == "UAH"


def test_jsonld_price_without_currency_resolved_by_matching_text_amount():
    """auto.ria pattern: JSON-LD has "price" but no priceCurrency; the text
    fallback finds the SAME amount currency-marked -> its currency is
    trusted. A mismatching text amount must NOT donate its currency."""
    html = (
        '<html><head><script type="application/ld+json">'
        '{"@type": "Product", "name": "Nissan Rogue 2014", "offers": {"price": "10000"}}'
        "</script></head><body>ціна 10\xa0000 $</body></html>"
    )
    facts = extract_advert_facts_from_html(html)
    assert facts["price"] == 10000.0
    assert facts["currency"] == "USD"

    html_mismatch = (
        '<html><head><script type="application/ld+json">'
        '{"@type": "Product", "name": "Nissan Rogue 2014", "offers": {"price": "10000"}}'
        "</script></head><body>розстрочка 250 $/міс</body></html>"
    )
    facts = extract_advert_facts_from_html(html_mismatch)
    assert facts["price"] == 10000.0
    assert facts["currency"] is None
