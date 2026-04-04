import pandas as pd
from pathlib import Path

from bianca_features import build_bianca_features_jigsaw

RAW_PATH = Path("/Users/biancaroscamayer/Downloads/jigsaw-dataset/train.csv")
OUT_PATH = Path("01_data/01_processed/train_bianca_features.csv")


def main():
    df = pd.read_csv(RAW_PATH)
    df = build_bianca_features_jigsaw(df, text_col="comment_text", use_spacy=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    print(df[[
        "id",
        "comment_text",
        "profanity_word_count",
        "obfuscated_profanity_count",
        "dependency_tuples",
        "abusive_dependency_tuple_count",
        "any_toxic"
    ]].head())


if __name__ == "__main__":
    main()