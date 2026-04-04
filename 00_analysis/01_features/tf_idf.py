from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')
X = df["comment_text"].tolist()
y = df["toxic"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

def get_tfidf_features(texts):
    # Word TF-IDF
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        stop_words='english',
        max_features=20000
    )

    # Character TF-IDF
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=30000
    )

    X_word = word_vectorizer.fit_transform(texts)
    X_char = char_vectorizer.fit_transform(texts)

    # Combine features
    X_combined = hstack([X_word, X_char])

    print("Shape of n-word-gram features:", X_word.shape)
    print("Shape of m-character-gram features:", X_char.shape)
    print("Shape of word and character grams combined features:", X_combined.shape)

    return X_combined

texts = X_train.copy()
combined_word_char_grams = get_tfidf_features(texts)