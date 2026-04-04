import re
from typing import List, Tuple, Optional

import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None


TEXT_COL = "comment_text"
ID_COL = "id"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

PROFANITY_LEXICON = {
    "ass", "asshole", "bastard", "bitch", "bullshit", "crap", "cunt",
    "damn", "dick", "dumbass", "fuck", "fucker", "fucking", "idiot",
    "jackass", "loser", "moron", "retard", "shit", "slut", "stupid",
    "trash", "whore"
}

LEETSPEAK_MAP = {
    "@": "a",
    "$": "s",
    "5": "s",
    "0": "o",
    "1": "i",
    "!": "i",
    "3": "e",
    "7": "t",
    "+": "t"
}

TARGET_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd"}
COMPLEMENT_DEPS = {"xcomp", "ccomp"}

ABUSIVE_VERBS = {
    "hate", "kill", "deserve", "shut", "die", "attack", "ban"
}

ABUSIVE_COMPLEMENTS = {
    "death", "trash", "idiot", "stupid", "pathetic", "worthless"
}


def safe_text(text) -> str:
    if pd.isna(text):
        return ""
    return str(text)


def tokenize_lower(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def profanity_word_count(text: str, profanity_lexicon: Optional[set] = None) -> int:
    if profanity_lexicon is None:
        profanity_lexicon = PROFANITY_LEXICON
    tokens = tokenize_lower(safe_text(text))
    return sum(1 for token in tokens if token in profanity_lexicon)


def normalize_leetspeak_token(token: str) -> str:
    token = token.lower()
    normalized = []
    for ch in token:
        normalized.append(LEETSPEAK_MAP.get(ch, ch))
    token = "".join(normalized)
    token = re.sub(r"[^a-z]", "", token)
    return token


def obfuscated_profanity_count(text: str, profanity_lexicon: Optional[set] = None) -> int:
    if profanity_lexicon is None:
        profanity_lexicon = PROFANITY_LEXICON

    text = safe_text(text)
    raw_tokens = text.split()
    count = 0

    for raw in raw_tokens:
        raw_clean_alpha = re.sub(r"[^a-z]", "", raw.lower())
        normalized = normalize_leetspeak_token(raw)

        if normalized in profanity_lexicon and raw_clean_alpha not in profanity_lexicon:
            count += 1

    return count


def load_spacy_model(model_name: str = "en_core_web_sm"):
    if spacy is None:
        raise ImportError("spaCy is not installed. Run: pip install spacy")
    try:
        return spacy.load(model_name)
    except OSError:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        )


def extract_dependency_tuples(text: str, nlp) -> List[Tuple[str, str, str]]:
    text = safe_text(text).strip()
    if not text:
        return []

    doc = nlp(text)
    tuples = []

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        subject = None
        obj_or_comp = None

        for child in token.children:
            if child.dep_ in TARGET_DEPS and subject is None:
                subject = child.text.lower()

            if child.dep_ in OBJECT_DEPS.union(COMPLEMENT_DEPS) and obj_or_comp is None:
                obj_or_comp = child.text.lower()

        if subject and obj_or_comp:
            tuples.append((subject, token.lemma_.lower(), obj_or_comp))

    return tuples


def dependency_tuples_as_text(text: str, nlp) -> str:
    tuples = extract_dependency_tuples(text, nlp)
    return "|".join([f"{s}_{v}_{o}" for s, v, o in tuples])


def abusive_dependency_tuple_count(text: str, nlp) -> int:
    tuples = extract_dependency_tuples(text, nlp)
    count = 0
    for subj, verb, obj in tuples:
        if verb in ABUSIVE_VERBS or obj in ABUSIVE_COMPLEMENTS:
            count += 1
    return count


def add_any_toxic_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    existing_labels = [c for c in LABEL_COLS if c in df.columns]
    if existing_labels:
        df["any_toxic"] = (df[existing_labels].sum(axis=1) > 0).astype(int)
    return df


def build_bianca_features_jigsaw(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    use_spacy: bool = True
) -> pd.DataFrame:
    df = df.copy()

    df["profanity_word_count"] = df[text_col].apply(profanity_word_count)
    df["obfuscated_profanity_count"] = df[text_col].apply(obfuscated_profanity_count)

    if use_spacy:
        nlp = load_spacy_model("en_core_web_sm")
        df["dependency_tuples"] = df[text_col].apply(lambda x: dependency_tuples_as_text(x, nlp))
        df["abusive_dependency_tuple_count"] = df[text_col].apply(lambda x: abusive_dependency_tuple_count(x, nlp))
    else:
        df["dependency_tuples"] = ""
        df["abusive_dependency_tuple_count"] = 0

    df = add_any_toxic_label(df)
    return df