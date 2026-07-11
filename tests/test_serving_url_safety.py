"""Tests for the SSRF safety checks that gate POST /predict-url
(serving/app.py: `validate_image_url`, `_resolve_and_validate_host`,
`_is_disallowed_ip`).

These exercise the *validation* layer only (no real network access, and no
real HTTP fetch) -- DNS resolution is mocked via monkeypatching
`socket.getaddrinfo`, matching how a real resolver call is shaped:
`[(family, type, proto, canonname, (address, port, ...)), ...]`.
"""

from __future__ import annotations

import socket

import pytest

from serving.app import (
    URLValidationError,
    _is_disallowed_ip,
    _resolve_and_validate_host,
    validate_image_url,
)


def _fake_getaddrinfo(addresses):
    """Build a fake `socket.getaddrinfo` returning the given IP address
    strings, shaped like the real stdlib return value closely enough for
    `_resolve_and_validate_host` to parse.
    """

    def _impl(host, port, *args, **kwargs):
        infos = []
        for addr in addresses:
            if ":" in addr:  # crude IPv4 vs IPv6 sniff, good enough for tests
                infos.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (addr, 0, 0, 0)))
            else:
                infos.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0)))
        return infos

    return _impl


def _mock_dns(monkeypatch, hostname_to_ips: dict[str, list[str]]) -> None:
    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host not in hostname_to_ips:
            raise socket.gaierror(f"no fake DNS entry for {host!r}")
        return _fake_getaddrinfo(hostname_to_ips[host])(host, port, *args, **kwargs)

    monkeypatch.setattr("serving.app.socket.getaddrinfo", fake_getaddrinfo)


# --- _is_disallowed_ip -------------------------------------------------


@pytest.mark.parametrize(
    "ip_str",
    [
        "127.0.0.1",  # loopback
        "10.0.0.1",  # private (RFC1918)
        "172.16.0.5",  # private (RFC1918)
        "192.168.1.1",  # private (RFC1918)
        "169.254.169.254",  # link-local / cloud metadata endpoint
        "224.0.0.1",  # multicast
        "0.0.0.0",  # unspecified
        "::1",  # IPv6 loopback
        "fe80::1",  # IPv6 link-local
        "fc00::1",  # IPv6 unique local (private)
    ],
)
def test_disallowed_ips_are_rejected(ip_str):
    import ipaddress

    assert _is_disallowed_ip(ipaddress.ip_address(ip_str)) is True


@pytest.mark.parametrize(
    "ip_str",
    [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",  # example.com-ish public IP
        "2606:4700:4700::1111",  # public IPv6 (Cloudflare DNS)
    ],
)
def test_public_ips_are_allowed(ip_str):
    import ipaddress

    assert _is_disallowed_ip(ipaddress.ip_address(ip_str)) is False


# --- validate_image_url: scheme / hostname checks -----------------------


@pytest.mark.parametrize("bad_url", ["ftp://example.com/x.jpg", "file:///etc/passwd", "gopher://example.com/x"])
def test_rejects_non_http_schemes(bad_url):
    with pytest.raises(URLValidationError):
        validate_image_url(bad_url)


def test_rejects_localhost_hostname():
    with pytest.raises(URLValidationError, match="localhost"):
        validate_image_url("http://localhost/x.jpg")


def test_rejects_url_with_no_hostname():
    with pytest.raises(URLValidationError):
        validate_image_url("http:///x.jpg")


# --- validate_image_url: literal-IP URLs --------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.jpg",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.1/x.jpg",
        "http://[::1]/x.jpg",
    ],
)
def test_rejects_literal_ip_urls_pointing_at_disallowed_ranges(url):
    with pytest.raises(URLValidationError):
        validate_image_url(url)


def test_accepts_literal_public_ip_url(monkeypatch):
    _mock_dns(monkeypatch, {"8.8.8.8": ["8.8.8.8"]})
    hostname, resolved = validate_image_url("http://8.8.8.8/x.jpg")
    assert hostname == "8.8.8.8"
    assert resolved == ["8.8.8.8"]


# --- validate_image_url / _resolve_and_validate_host: DNS resolution ----


def test_accepts_public_hostname_resolving_to_public_ip(monkeypatch):
    _mock_dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    hostname, resolved = validate_image_url("https://example.com/car.jpg")
    assert hostname == "example.com"
    assert resolved == ["93.184.216.34"]


def test_rejects_hostname_resolving_to_private_ip(monkeypatch):
    """Simulates an attacker-controlled DNS name that resolves to an
    internal address (e.g. a rebinding attempt or a misconfigured internal
    hostname)."""
    _mock_dns(monkeypatch, {"evil.example.com": ["10.0.0.5"]})
    with pytest.raises(URLValidationError):
        validate_image_url("http://evil.example.com/x.jpg")


def test_rejects_hostname_if_any_resolved_address_is_disallowed(monkeypatch):
    """Even if one A record is public, a single disallowed address among
    several must still reject the whole host."""
    _mock_dns(monkeypatch, {"multi.example.com": ["93.184.216.34", "127.0.0.1"]})
    with pytest.raises(URLValidationError):
        validate_image_url("http://multi.example.com/x.jpg")


def test_rejects_hostname_that_fails_to_resolve(monkeypatch):
    _mock_dns(monkeypatch, {})  # nothing resolves
    with pytest.raises(URLValidationError):
        validate_image_url("http://nonexistent.invalid/x.jpg")


def test_resolve_and_validate_host_returns_all_public_addresses(monkeypatch):
    _mock_dns(monkeypatch, {"multi-public.example.com": ["93.184.216.34", "1.1.1.1"]})
    addresses = _resolve_and_validate_host("multi-public.example.com")
    assert set(addresses) == {"93.184.216.34", "1.1.1.1"}
