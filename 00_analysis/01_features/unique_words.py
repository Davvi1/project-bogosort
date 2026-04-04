"""Unique-token ratio per row. Same input prep as ``text_features``.
``unique_word_ratio`` (float) = unique tokens / total tokens (``split()``); case- and punctuation-sensitive; empty -> 0."""

import pandas as pd
import unittest


def unique_word_ratio(text: str) -> float:
    """Return number of unique words divided by total words."""
    # Split by whitespace into tokens.
    words = text.split()
    if len(words) == 0:
        return 0.0

    # set(...) removes duplicates.
    unique_words = set(words)
    return len(unique_words) / len(words)


def add_unique_word_feature(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Add unique-word ratio feature to a DataFrame."""
    # Validate required input column.
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Keep original input unchanged.
    out = df.copy()
    # Make sure every row is a string.
    out[text_column] = out[text_column].fillna("").astype(str)
    out["unique_word_ratio"] = out[text_column].apply(unique_word_ratio)
    return out


class TestUniqueWords(unittest.TestCase):
    """Basic regression tests for unique word features."""

    def test_unique_word_ratio_empty(self):
        self.assertEqual(unique_word_ratio(""), 0.0)

    def test_unique_word_ratio_basic(self):
        self.assertEqual(unique_word_ratio("a b a c"), 0.75)

    def test_add_unique_word_feature_column_exists(self):
        df = pd.DataFrame({"text": ["a a a", "one two three"]})
        out = add_unique_word_feature(df)
        self.assertIn("unique_word_ratio", out.columns)

    def test_add_unique_word_feature_values(self):
        df = pd.DataFrame({"text": ["a a a", None, "x y"]})
        out = add_unique_word_feature(df)
        self.assertEqual(out.loc[0, "unique_word_ratio"], 1 / 3)
        self.assertEqual(out.loc[1, "unique_word_ratio"], 0.0)
        self.assertEqual(out.loc[2, "unique_word_ratio"], 1.0)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other": ["text"]})
        with self.assertRaises(ValueError):
            add_unique_word_feature(df, text_column="text")


if __name__ == "__main__":
    unittest.main()
