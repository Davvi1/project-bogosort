"""
Feature engineering functions for toxicity classification.

Three feature domains:
  1. Sentiment         -- VADER-based scores
  2. Second-person     -- pronoun presence and density
  3. Identity groups   -- mentions of protected/demographic groups

Each domain exposes a single extract_* function that accepts a raw comment
string and returns a flat dict of {feature_name: value}.

An extract_all() convenience wrapper merges all three dicts.

Usage:
    from features import extract_all
    features = extract_all("You are a total idiot")
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ===========================================================================
# 1. SENTIMENT-ANALYSIS  (VADER)
# ===========================================================================
#
# VADER returns four raw scores:
#   neg      -- proportion of tokens carrying negative sentiment  (0–1)
#   neu      -- proportion neutral                                (0–1)
#   pos      -- proportion carrying positive sentiment            (0–1)
#   compound -- normalised weighted composite                     (−1 to +1)
#
# We drop `neu` because neg + pos + neu == 1, so it carries no extra info.
# From the remaining three we derive two additional features:
#   vader_is_negative   -- hard binary flag (compound < −0.05)
#   vader_intensity     -- abs(compound); emotional strength regardless of sign
#   vader_pos_minus_neg -- net positivity ratio (pos − neg); negative = more negative sentiment
#
# Performance note: SentimentIntensityAnalyzer() loads a lexicon from disk and
# is expensive to construct. Using it as a default argument means Python creates
# it exactly once when the function is defined -- not on every call. The _sia
# parameter is an implementation detail; callers should never pass it.
# ---------------------------------------------------------------------------

def extract_sentiment(text: str, _sia=SentimentIntensityAnalyzer()) -> dict:
    """
    Extract VADER-based sentiment features from a comment.

    Parameters
    ----------
    text : str
        Raw comment text (no pre-processing required; VADER handles
        capitalisation, punctuation, and contractions internally).

    Returns
    -------
    dict with keys:
        vader_compound      float  −1 to +1   primary signal
        vader_neg           float  0 to 1     negative-token proportion
        vader_pos           float  0 to 1     positive-token proportion
        vader_is_negative   int    0 or 1     compound < −0.05
        vader_intensity     float  0 to 1     abs(compound)
        vader_pos_minus_neg float  −1 to 1    net positivity; negative values = more negative sentiment
    """
    scores = _sia.polarity_scores(text)
    compound = scores["compound"]
    neg = scores["neg"]
    pos = scores["pos"]

    return {
        "vader_compound":      compound,
        "vader_neg":           neg,
        "vader_pos":           pos,
        "vader_is_negative":   int(compound < -0.05), # Maybe we throw out, unless we do logistic regression
        "vader_intensity":     abs(compound),
        "vader_pos_minus_neg": round(pos - neg, 6), # Similar to compound, but ignores certain adjustments through how VADER works. Maybe we toss due to multicollinearity
    }


# ===========================================================================
# 2. SECOND-PERSON PRONOUNS
# ===========================================================================
#
# I thought of using something like an external library, but I read that
# it could result in very slow code, and Regex works quickly
# 
# ---------------------------------------------------------------------------

def extract_second_person_feature(
    text: str,                          
    _pattern=re.compile(                # I use re.compile(). Think this defines it once, rather than thousands of times.
        r"\b(you|your|yours|yourself|yourselves"  
        r"|you're|you'll|you've|you'd"            
        r"|ur||u r| u)\b",                         
        re.IGNORECASE,                 
    ),
) -> dict:                              
    """
    Extract second-person pronoun features from a comment.

    Parameters
    ----------
    text : str
        Raw comment text.

    Returns
    -------
    dict with keys:
        has_second_person      int    0 or 1
        second_person_count    int    >= 0
        second_person_density  float  count / word_count
    """
    matches = _pattern.findall(text)            
    count = len(matches)                        # how many second-person tokens were found in total
    word_count = max(len(text.split()), 1)      # total words in comment; max(..., 1) prevents division by zero on empty strings

    return {
        "has_second_person":     int(count > 0),            
        "second_person_count":   count,                      # Not sure how valuable, since it is dependent on comment length, but maybe could still offer info
        "second_person_density": round(count / word_count, 6), 
    }




# ===========================================================================
# 3. IDENTITY GROUP MENTIONS
# ===========================================================================
#
# WHY A CURATED REGEX LIST (not a library):
#   - No off-the-shelf NLP library covers this domain adequately.
#     spaCy NER identifies named entities ("Barack Obama"), not group terms.
#   - External APIs (Perspective) add latency and cost.
#   - A hand-curated list is transparent, auditable, and easily extended.
#
# TWO TIERS:
#   neutral_terms  -- words that can appear in toxic OR benign contexts
#                     ("black", "gay", "Muslim" appear in millions of neutral
#                     sentences; presence alone is a weak signal)
#   slur_terms     -- words that are nearly always derogatory in context
#                     (much stronger signal; kept separate so models can
#                     weight them differently)
#
# LEETSPEAK:
#   Common obfuscations (n1gger, f4ggot, r3tard) are handled via character
#   class alternatives in the regex patterns.
# ---------------------------------------------------------------------------

def extract_identity(
    text: str,
    _category_patterns=None,
    _all_identity_re=None,
    _slur_re=None,
) -> dict:
    """
    Extract identity group mention features from a comment.

    Parameters
    ----------
    text : str
        Raw comment text.

    Returns
    -------
    dict with keys:
        has_identity_mention   int    0 or 1   any term (neutral or slur)
        identity_mention_count int    >= 0     total matches
        has_identity_slur      int    0 or 1   slur present
        identity_slur_count    int    >= 0     total slur matches
        identity_race          int    0 or 1   race/ethnicity category
        identity_gender        int    0 or 1   gender category
        identity_sexuality     int    0 or 1   sexuality category
        identity_religion      int    0 or 1   religion category
        identity_disability    int    0 or 1   disability category
    """
    # Initialise compiled patterns on first call and cache them in the
    # function's default argument slots (same once-only principle as above,
    # but done lazily because the dict structure makes the inline definition
    # too unwieldy to read).
    if _category_patterns is None:
        _IDENTITY_CATEGORIES = {
            "race": [
                "black", "white", "asian", "hispanic", "latino", "latina",
                "latinx", "african", "arab", "jewish", "jew", "chinese",
                "mexican", "indian",
            ],
            "gender": [
                "woman", "women", "female", "male",
                "transgender", "trans", "nonbinary",
            ],
            "sexuality": [
                "gay", "lesbian", "bisexual", "queer", "homosexual", "lgbt",
                "lgbtq", "straight",
            ],
            "religion": [
                "muslim", "islam", "islamic", "christian", "christianity",
                "catholic", "jewish", "hindu", "buddhist", "atheist",
                "atheism", "sikh", "antisemit",
            ],
            "disability": [
                "disabled", "disability", "autistic", "autism",
                "blind", "deaf",
            ],
        }

        _SLUR_PATTERNS = [
            r"n[i1][g9][g9][e3]r",
            r"n[i1][g9][g9][a4]",
            r"ch[i1]nk",
            r"sp[i1][ck]",
            r"k[i1]ke",
            r"w[e3]tb[a4]ck",
            r"f[a4][g9][g9][o0]t",
            r"f[a4][g9]",
            r"dyke",
            r"tr[a4]nn[yi1]",
            r"r[e3]t[a4]rd",
            r"cr[i1]ppl[e3]",
            r"[s5]p[a4][s5]t[i1]c",
        ]

        _category_patterns = {
            cat: re.compile(
                r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
                re.IGNORECASE,
            )
            for cat, terms in _IDENTITY_CATEGORIES.items()
        }

        _all_identity_re = re.compile(
            r"\b(" + "|".join(
                re.escape(t)
                for terms in _IDENTITY_CATEGORIES.values()
                for t in terms
            ) + r")\b",
            re.IGNORECASE,
        )

        _slur_re = re.compile(
            r"\b(" + "|".join(_SLUR_PATTERNS) + r")\b",
            re.IGNORECASE,
        )

        # Cache back into the function's defaults so the next call skips this block
        extract_identity.__defaults__ = (
            _category_patterns,
            _all_identity_re,
            _slur_re,
        )

    neutral_matches = _all_identity_re.findall(text)
    slur_matches = _slur_re.findall(text)
    total_matches = len(neutral_matches) + len(slur_matches)

    category_flags = {
        f"identity_{cat}": int(bool(pat.search(text)))
        for cat, pat in _category_patterns.items()
    }

    return {
        "has_identity_mention":   int(total_matches > 0),
        "identity_mention_count": total_matches,
        "has_identity_slur":      int(len(slur_matches) > 0),
        "identity_slur_count":    len(slur_matches),
        **category_flags,
    }


# ===========================================================================
# Convenience: extract all features at once
# ===========================================================================
