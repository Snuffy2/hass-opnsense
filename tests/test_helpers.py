"""Unit tests for custom_components.opnsense.helpers."""

from custom_components.opnsense import helpers


def test_dict_get_basic_nested_and_missing():
    """dict_get returns nested values and falls back to defaults for missing paths."""
    data = {"a": {"b": {"c": 1}}, "x": None}
    # existing nested path
    assert helpers.dict_get(data, "a.b.c") == 1
    # missing path returns default (None)
    assert helpers.dict_get(data, "a.b.missing") is None
    # explicitly provided default is returned
    assert helpers.dict_get(data, "a.b.missing", default=5) == 5


def test_dict_get_numeric_key_on_mapping_and_non_mapping():
    """dict_get correctly parses numeric path segments into integer keys."""
    # mapping that actually uses integer keys: path '1' becomes int(1)
    data_with_int_keys = {1: "one", "nested": {2: "two"}}
    assert helpers.dict_get(data_with_int_keys, "1") == "one"
    # nested numeric key within mapping
    assert helpers.dict_get({"outer": {2: "two"}}, "outer.2") == "two"


def test_dict_get_non_mapping_and_defaults():
    """dict_get returns default when intermediate path elements are missing."""
    # when intermediate result is not a mapping/list the function should return default
    assert helpers.dict_get({}, "no.such.path") is None


def test_is_private_ip_ipv4_and_hostname_and_malformed():
    """is_private_ip identifies private IPs and handles hostnames/malformed URLs."""
    # private IPv4 addresses should return True
    assert helpers.is_private_ip("http://192.168.1.1") is True
    assert helpers.is_private_ip("https://10.0.0.5/path") is True

    # public IP should return False
    assert helpers.is_private_ip("http://8.8.8.8") is False

    # hostname (non-IP) should return False (ValueError branch)
    assert helpers.is_private_ip("https://example.com") is False

    # malformed URL or missing hostname -> False
    assert helpers.is_private_ip("not a url") is False
    assert helpers.is_private_ip("http:///nohost") is False
