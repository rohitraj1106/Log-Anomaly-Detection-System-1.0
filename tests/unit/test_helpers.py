"""
Unit Tests — Helper Utilities.

Tests timing, hashing, timestamp parsing, and data manipulation helpers.
"""

from datetime import datetime

import pytest

from utils.helpers import (
    chunk_list,
    compute_hash,
    ensure_directory,
    flatten_dict,
    parse_timestamp,
    safe_divide,
    truncate_string,
)


class TestComputeHash:
    """Tests for SHA-256 hashing utility."""

    @pytest.mark.unit
    def test_deterministic(self):
        """Same input always produces the same hash."""
        assert compute_hash("hello") == compute_hash("hello")

    @pytest.mark.unit
    def test_different_inputs(self):
        """Different inputs produce different hashes."""
        assert compute_hash("hello") != compute_hash("world")

    @pytest.mark.unit
    def test_returns_hex_string(self):
        """Output is a 64-character hex string (SHA-256)."""
        h = compute_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    @pytest.mark.unit
    def test_empty_string(self):
        """Empty string produces a valid hash."""
        h = compute_hash("")
        assert len(h) == 64


class TestSafeDivide:
    """Tests for safe division utility."""

    @pytest.mark.unit
    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    @pytest.mark.unit
    def test_zero_denominator_returns_default(self):
        assert safe_divide(10, 0) == 0.0

    @pytest.mark.unit
    def test_zero_denominator_custom_default(self):
        assert safe_divide(10, 0, default=-1.0) == -1.0

    @pytest.mark.unit
    def test_zero_numerator(self):
        assert safe_divide(0, 10) == 0.0


class TestParseTimestamp:
    """Tests for multi-format timestamp parser."""

    @pytest.mark.unit
    def test_standard_format(self):
        result = parse_timestamp("2024-01-15 10:30:15")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1

    @pytest.mark.unit
    def test_iso_format(self):
        result = parse_timestamp("2024-01-15T10:30:15")
        assert isinstance(result, datetime)

    @pytest.mark.unit
    def test_invalid_format_returns_none(self):
        result = parse_timestamp("not-a-timestamp")
        assert result is None


class TestChunkList:
    """Tests for list chunking utility."""

    @pytest.mark.unit
    def test_even_chunks(self):
        result = chunk_list([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    @pytest.mark.unit
    def test_uneven_chunks(self):
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    @pytest.mark.unit
    def test_empty_list(self):
        result = chunk_list([], 3)
        assert result == []


class TestFlattenDict:
    """Tests for dict flattening utility."""

    @pytest.mark.unit
    def test_flat_dict_unchanged(self):
        result = flatten_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    @pytest.mark.unit
    def test_nested_dict(self):
        result = flatten_dict({"a": {"b": 1, "c": 2}})
        assert result == {"a.b": 1, "a.c": 2}

    @pytest.mark.unit
    def test_deeply_nested(self):
        result = flatten_dict({"a": {"b": {"c": 3}}})
        assert result == {"a.b.c": 3}


class TestTruncateString:
    """Tests for string truncation."""

    @pytest.mark.unit
    def test_short_string_unchanged(self):
        assert truncate_string("hello", 10) == "hello"

    @pytest.mark.unit
    def test_long_string_truncated(self):
        result = truncate_string("a" * 600, 500)
        assert len(result) == 500
        assert result.endswith("...")

    @pytest.mark.unit
    def test_exact_length(self):
        s = "a" * 500
        assert truncate_string(s, 500) == s


class TestEnsureDirectory:
    """Tests for directory creation helper."""

    @pytest.mark.unit
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "sub" / "dir"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()

    @pytest.mark.unit
    def test_existing_directory_no_error(self, tmp_path):
        result = ensure_directory(tmp_path)
        assert result.exists()
