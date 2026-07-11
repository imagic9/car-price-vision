"""Tests for the advert/listing-page photo extraction and multi-photo
aggregation added to POST /predict-url (serving/app.py:
`extract_photo_urls_from_html`, `_aggregate_photo_predictions`).

These are pure unit tests: no real network access, and no model inference --
`extract_photo_urls_from_html` only parses inline HTML fixtures, and
`_aggregate_photo_predictions` only does arithmetic over hand-built
(year_z, log_price_z) tuples. See tests/test_serving_url_safety.py for the
SSRF-gate tests these build on (every extracted URL still has to pass that
same gate before being fetched -- not re-tested here since it's identical
code).
"""

from __future__ import annotations

from serving.app import (
    MAX_PHOTOS_PER_PAGE,
    _aggregate_photo_predictions,
    extract_photo_urls_from_html,
)

# --- extract_photo_urls_from_html --------------------------------------


def test_extracts_og_image():
    html = b"""
    <html><head>
      <meta property="og:image" content="https://example.com/photos/car1.jpg">
    </head><body></body></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/123")
    assert urls == ["https://example.com/photos/car1.jpg"]


def test_extracts_og_image_secure_url_and_twitter_image_in_priority_order():
    html = b"""
    <html><head>
      <meta property="og:image:secure_url" content="https://example.com/a.jpg">
      <meta name="twitter:image" content="https://example.com/b.jpg">
    </head><body>
      <img src="https://example.com/c.jpg">
    </body></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/123")
    # og:image family first, then twitter:image, then <img> as last resort.
    assert urls == [
        "https://example.com/a.jpg",
        "https://example.com/b.jpg",
        "https://example.com/c.jpg",
    ]


def test_extracts_jsonld_image_array():
    html = b"""
    <html><head>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "name": "2016 Ford Focus",
        "image": ["https://example.com/ld1.jpg", "https://example.com/ld2.jpg"]
      }
      </script>
    </head><body></body></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/123")
    assert urls == ["https://example.com/ld1.jpg", "https://example.com/ld2.jpg"]


def test_extracts_jsonld_image_string_and_list_of_objects():
    html = b"""
    <html><head>
      <script type="application/ld+json">
      [
        {"@type": "Product", "image": "https://example.com/single.jpg"},
        {"@type": "Organization", "name": "Some Dealer"}
      ]
      </script>
    </head></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/123")
    assert urls == ["https://example.com/single.jpg"]


def test_malformed_jsonld_does_not_crash_and_is_skipped():
    html = b"""
    <html><head>
      <script type="application/ld+json">
      { this is not valid json at all ][
      </script>
      <meta property="og:image" content="https://example.com/fallback.jpg">
    </head></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/123")
    assert urls == ["https://example.com/fallback.jpg"]


def test_relative_urls_resolved_against_base_url():
    html = b"""
    <html><head>
      <meta property="og:image" content="/media/photos/car1.jpg">
    </head><body>
      <img data-src="../img/car2.jpg">
    </body></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://auto.example.com/en/listing/42/")
    assert urls == [
        "https://auto.example.com/media/photos/car1.jpg",
        "https://auto.example.com/en/listing/img/car2.jpg",
    ]


def test_logo_icon_sprite_avatar_urls_are_filtered():
    html = b"""
    <html><head>
      <meta property="og:image" content="https://example.com/site-logo.png">
    </head><body>
      <img src="https://example.com/nav-icon.png">
      <img src="https://example.com/sprite-sheet.png">
      <img src="https://example.com/user-avatar.jpg">
      <img src="https://example.com/photos/real-car.jpg">
    </body></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/1")
    assert urls == ["https://example.com/photos/real-car.jpg"]


def test_non_http_scheme_and_unparseable_urls_are_dropped():
    html = b"""
    <html><head>
      <meta property="og:image" content="data:image/png;base64,abcd">
    </head><body>
      <img src="javascript:void(0)">
      <img src="https://example.com/ok.jpg">
    </body></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/1")
    assert urls == ["https://example.com/ok.jpg"]


def test_duplicate_urls_are_deduped_preserving_order():
    html = b"""
    <html><head>
      <meta property="og:image" content="https://example.com/car.jpg">
    </head><body>
      <img src="https://example.com/car.jpg">
      <img src="https://example.com/other.jpg">
    </body></html>
    """
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/1")
    assert urls == ["https://example.com/car.jpg", "https://example.com/other.jpg"]


def test_capped_at_max_photos_per_page():
    imgs = "".join(f'<img src="https://example.com/img{i}.jpg">' for i in range(20))
    html = f"<html><body>{imgs}</body></html>".encode()
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/1")
    assert len(urls) == MAX_PHOTOS_PER_PAGE
    assert urls == [f"https://example.com/img{i}.jpg" for i in range(MAX_PHOTOS_PER_PAGE)]


def test_malformed_html_does_not_crash():
    html = b"<html><head><meta property=og:image content=https://example.com/a.jpg><body><img src='https://example.com/b.jpg'"
    # Truncated/unclosed tags -- html.parser tolerates this; just assert no
    # exception propagates and we still get a sensible-ish result.
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/1")
    assert isinstance(urls, list)


def test_completely_empty_or_no_images_returns_empty_list():
    html = b"<html><head><title>No car photos here</title></head><body><p>Nothing.</p></body></html>"
    urls = extract_photo_urls_from_html(html, base_url="https://example.com/listing/1")
    assert urls == []


# --- _aggregate_photo_predictions ---------------------------------------


def test_aggregate_median_odd_count():
    z_vectors = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    median_year_z, median_log_price_z, rep_idx = _aggregate_photo_predictions(z_vectors)
    assert median_year_z == 1.0
    assert median_log_price_z == 2.0
    assert rep_idx == 1  # (1.0, 2.0) is exactly the median vector


def test_aggregate_median_even_count():
    z_vectors = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    median_year_z, median_log_price_z, rep_idx = _aggregate_photo_predictions(z_vectors)
    # statistics.median averages the two middle values for an even-length list.
    assert median_year_z == 1.5
    assert median_log_price_z == 1.5
    # Both index 1 (1.0, 1.0) and index 2 (2.0, 2.0) are equidistant from
    # (1.5, 1.5); min() picks the first one encountered.
    assert rep_idx in (1, 2)


def test_aggregate_outlier_interior_shot_does_not_move_the_median():
    # Five near-consistent exterior-shot predictions plus one wild outlier
    # (e.g. an interior/dashboard shot that's out-of-distribution for this
    # exterior-trained model) -- the median should track the cluster, not
    # the outlier, which is exactly the point of using median over mean.
    cluster = [(1.0, 2.0), (1.05, 2.02), (0.95, 1.98), (1.02, 2.01), (0.98, 1.99)]
    outlier = (50.0, -80.0)
    z_vectors = cluster + [outlier]

    median_year_z, median_log_price_z, rep_idx = _aggregate_photo_predictions(z_vectors)

    assert abs(median_year_z - 1.0) < 0.1
    assert abs(median_log_price_z - 2.0) < 0.1
    # The representative photo must be one of the clustered ones, not the outlier.
    assert rep_idx != len(z_vectors) - 1

    # Contrast: a plain mean *would* be dragged noticeably off the cluster.
    mean_year_z = sum(z[0] for z in z_vectors) / len(z_vectors)
    assert mean_year_z > 8.0  # outlier visibly drags the mean, unlike the median


def test_aggregate_representative_is_closest_to_median_by_l2():
    z_vectors = [(0.0, 0.0), (10.0, 10.0), (0.4, 0.3)]
    median_year_z, median_log_price_z, rep_idx = _aggregate_photo_predictions(z_vectors)
    assert median_year_z == 0.4
    assert median_log_price_z == 0.3
    assert rep_idx == 2  # (0.4, 0.3) is exactly the median vector, distance 0


def test_aggregate_single_photo_is_its_own_representative():
    z_vectors = [(1.23, -0.45)]
    median_year_z, median_log_price_z, rep_idx = _aggregate_photo_predictions(z_vectors)
    assert median_year_z == 1.23
    assert median_log_price_z == -0.45
    assert rep_idx == 0
