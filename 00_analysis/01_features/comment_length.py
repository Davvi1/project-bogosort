"""Comment length. Input: ``text_column``; NaN -> ""; ``str``. No ``strip``.
Outputs: ``comment_length_chars`` (``len``), ``comment_length_words`` (``split()`` count)."""

import pandas as pd
import unittest


def comment_length_chars(text: str) -> int:
    """Return character length of text."""
    return len(text)


def comment_length_words(text: str) -> int:
    """Return number of whitespace-separated words."""
    words = text.split()
    return len(words)


def add_comment_length_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Add comment length features (chars and words)."""
    # Validate required input column.
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Keep original input unchanged.
    out = df.copy()
    # Convert to string and replace missing values.
    out[text_column] = out[text_column].fillna("").astype(str)

    # Character-level length.
    out["comment_length_chars"] = out[text_column].apply(comment_length_chars)
    # Token-level length.
    out["comment_length_words"] = out[text_column].apply(comment_length_words)
    return out


class TestCommentLength(unittest.TestCase):
    """Basic regression tests for comment length features."""

    def test_comment_length_chars(self):
        self.assertEqual(comment_length_chars("Hi!!"), 4)
        self.assertEqual(comment_length_chars(""), 0)

    def test_comment_length_words(self):
        self.assertEqual(comment_length_words("one two three"), 3)
        self.assertEqual(comment_length_words(""), 0)

    def test_add_comment_length_features_columns_exist(self):
        df = pd.DataFrame({"text": ["Hi!!", "one two", None]})
        out = add_comment_length_features(df)
        self.assertIn("comment_length_chars", out.columns)
        self.assertIn("comment_length_words", out.columns)

    def test_add_comment_length_features_values(self):
        df = pd.DataFrame({"text": ["Hi!!", "one two"]})
        out = add_comment_length_features(df)
        self.assertEqual(out.loc[0, "comment_length_chars"], 4)
        self.assertEqual(out.loc[1, "comment_length_words"], 2)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other": ["text"]})
        with self.assertRaises(ValueError):
            add_comment_length_features(df, text_column="text")


if __name__ == "__main__":
    unittest.main()
