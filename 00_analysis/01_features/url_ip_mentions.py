"""URL / IPv4 mention counts (heuristic). Input: ``text_column``; NaN -> ""; ``str``.
URLs: ``http(s)://…`` or ``www.…`` until whitespace (case-insensitive). IPv4 only; no IPv6.
Out: ``url_mention_count``, ``ip_mention_count``, ``mentions_url_or_ip`` (0/1). Plain ``example.com`` not matched."""

import re
import unittest

import pandas as pd

# http(s)://... or www.... 
_URL_PATTERN = re.compile(
    r"(?:https?://[^\s]+)|(www\.[^\s]+)",
    flags=re.IGNORECASE,
)

# IPv4 octets 0–255 
_IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
)


def url_mention_count(text: str) -> int:
    """Count substrings that look like URLs (http(s) or www.)."""
    if not text:
        return 0
    return len(_URL_PATTERN.findall(text))


def ip_mention_count(text: str) -> int:
    """Count IPv4-like tokens in text."""
    if not text:
        return 0
    return len(_IPV4_PATTERN.findall(text))


def mentions_url_or_ip(text: str) -> int:
    """Binary flag: 1 if at least one URL-like or IPv4 mention, else 0."""
    return 1 if (url_mention_count(text) > 0 or ip_mention_count(text) > 0) else 0


def add_url_ip_mention_features(
    df: pd.DataFrame, text_column: str = "text"
) -> pd.DataFrame:
    """
    Add columns:
      - url_mention_count
      - ip_mention_count
      - mentions_url_or_ip (0/1)
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    out = df.copy()
    out[text_column] = out[text_column].fillna("").astype(str)

    out["url_mention_count"] = out[text_column].apply(url_mention_count)
    out["ip_mention_count"] = out[text_column].apply(ip_mention_count)
    out["mentions_url_or_ip"] = out[text_column].apply(mentions_url_or_ip)
    return out


class TestUrlIpMentions(unittest.TestCase):
    """Regression tests for URL / IP mention features."""

    def test_url_mention_count(self):
        self.assertEqual(url_mention_count(""), 0)
        self.assertEqual(url_mention_count("see https://example.com/path"), 1)
        self.assertEqual(url_mention_count("a http://a.com b www.foo.de x"), 2)

    def test_ip_mention_count(self):
        self.assertEqual(ip_mention_count(""), 0)
        self.assertEqual(ip_mention_count("connect 192.168.0.1 now"), 1)
        self.assertEqual(ip_mention_count("10.0.0.1 and 10.0.0.2"), 2)

    def test_mentions_url_or_ip(self):
        self.assertEqual(mentions_url_or_ip("plain text"), 0)
        self.assertEqual(mentions_url_or_ip("go to https://x.com"), 1)
        self.assertEqual(mentions_url_or_ip("ping 127.0.0.1"), 1)

    def test_add_columns(self):
        df = pd.DataFrame({"text": ["ok", "http://a.de", None]})
        out = add_url_ip_mention_features(df)
        self.assertIn("url_mention_count", out.columns)
        self.assertIn("ip_mention_count", out.columns)
        self.assertIn("mentions_url_or_ip", out.columns)
        self.assertEqual(out.loc[1, "url_mention_count"], 1)
        self.assertEqual(out.loc[2, "mentions_url_or_ip"], 0)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other": ["x"]})
        with self.assertRaises(ValueError):
            add_url_ip_mention_features(df, text_column="text")


if __name__ == "__main__":
    unittest.main()
