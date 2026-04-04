"""Row-wise text stats. Input: ``text_column``; NaN -> ""; then ``str``.
Words = ``split()`` on whitespace. Outputs: ``char_count`` (int), ``exclamation_count`` (int),
``upper_ratio`` (float, all-caps tokens with len>=2 / token count; empty -> 0)."""

import pandas as pd
import unittest


def uppercase_ratio(text: str) -> float:
    """Return share of fully uppercase words among all words."""
    # Split text into whitespace-separated tokens.
    words = text.split()
    if len(words) == 0:
        return 0.0

    # Count tokens that are fully uppercase and at least 2 chars long.
    upper_words = [w for w in words if w.isupper() and len(w) > 1]
    return len(upper_words) / len(words)


def add_text_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Add character, exclamation, and uppercase-ratio features."""
    # Validate required input column.
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Work on a copy to avoid mutating caller DataFrame.
    out = df.copy()
    # Normalize missing values so string operations always work.
    out[text_column] = out[text_column].fillna("").astype(str)

    # Total number of characters.
    out["char_count"] = out[text_column].apply(len)

    # Number of exclamation marks.
    out["exclamation_count"] = out[text_column].apply(lambda x: x.count("!"))

    # Share of uppercase words.
    out["upper_ratio"] = out[text_column].apply(uppercase_ratio)

    return out


class TestTextFeatures(unittest.TestCase):
    """Basic regression tests for text feature helpers."""

    def test_uppercase_ratio_empty(self):
        self.assertEqual(uppercase_ratio(""), 0.0)

    def test_uppercase_ratio_basic(self):
        self.assertEqual(uppercase_ratio("HELLO world TEST me"), 0.5)

    def test_add_text_features_columns_exist(self):
        df = pd.DataFrame({"text": ["Hi!!", "HELLO world", None]})
        out = add_text_features(df)
        self.assertIn("char_count", out.columns)
        self.assertIn("exclamation_count", out.columns)
        self.assertIn("upper_ratio", out.columns)

    def test_add_text_features_values(self):
        df = pd.DataFrame({"text": ["Hi!!", "HELLO world"]})
        out = add_text_features(df)
        self.assertEqual(out.loc[0, "char_count"], 4)
        self.assertEqual(out.loc[0, "exclamation_count"], 2)
        self.assertEqual(out.loc[1, "upper_ratio"], 0.5)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other": ["text"]})
        with self.assertRaises(ValueError):
            add_text_features(df, text_column="text")


if __name__ == "__main__":
    unittest.main()
