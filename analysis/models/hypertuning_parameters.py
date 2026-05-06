"""
Hyperparameter Tuning
=============================================================
Tunes LassoLogisticRegression on the Jigsaw toxicity dataset.

EDA-informed design decisions
------------------------------
  EDA-2: ~9:1 class imbalance → PR-AUC as primary metric, balanced
         sample weights passed through fit_params.
  EDA-4: Multicollinearity in VADER cluster → wide alpha range so Lasso
         can suppress redundant features without nuking sparse signals.
  EDA-4: Top signals expected: vader_neg, second_person_density,
         profanity_count, uppercase_ratio → sparsity report confirms this.
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from analysis.models.data_pipeline import DataPipeline
from analysis.models.lasso import LassoLogisticRegression
from analysis.features.build_features import FeatureBuilder, FeaturePreprocessor

ARTIFACTS_DIR = "./analysis/models/artifacts"
TARGET_COL    = "toxic"
N_FOLDS       = 5

PARAM_GRID = {
    "alpha":         [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "max_iter":      [1000, 2000],
}


# ── Feature prep ──────────────────────────────────────────────────────────────
# FeatureBuilder manages its own TF-IDF state so it can't live inside a
# sklearn Pipeline. We fit it once on full X_train, then transform per fold.

def build_features(X_train: pd.Series, X_val: pd.Series = None):
    """
    Fit FeatureBuilder on X_train, transform both splits.
    Returns numpy arrays ready for StandardScaler → LassoLogisticRegression.
    """
    fb = FeatureBuilder()

    if os.path.exists(fb.tfidf_path):
        fb.load()
    else:
        fb.fit(X_train)

    X_train_feat = fb.transform(X_train, split="train")

    preprocessor = FeaturePreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_feat)

    if X_val is not None:
        X_val_feat = fb.transform(X_val, split="test")
        X_val_proc = preprocessor.transform(X_val_feat)
        return X_train_proc, X_val_proc

    return X_train_proc, None


def build_sklearn_pipeline(alpha: float, learning_rate: float, max_iter: int) -> Pipeline:
    """
    StandardScaler → LassoLogisticRegression.
    FeatureBuilder/FeaturePreprocessor run before this (see manual CV loop).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LassoLogisticRegression(
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
        )),
    ])


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_fold(pipe, X_tr, y_tr, X_val, y_val, sample_weight):
    pipe.fit(X_tr, y_tr, clf__sample_weight=sample_weight)

    y_proba = pipe.predict_proba(X_val)[:, 1]
    y_pred  = pipe.predict(X_val)
    y_proba_tr = pipe.predict_proba(X_tr)[:, 1]

    return {
        "pr_auc":    average_precision_score(y_val, y_proba),
        "roc_auc":   roc_auc_score(y_val, y_proba),
        "macro_f1":  f1_score(y_val, y_pred, average="macro", zero_division=0),
        "pr_auc_tr": average_precision_score(y_tr, y_proba_tr),
    }


# ── Grid search ───────────────────────────────────────────────────────────────

def tune(X_train: pd.Series, y_train: np.ndarray) -> pd.DataFrame:
    """
    Manual stratified k-fold grid search.

    FeatureBuilder can't live inside a sklearn Pipeline (it manages its own
    TF-IDF state), so we pre-transform each fold manually then run
    StandardScaler → LassoLogisticRegression as the sklearn Pipeline.
    """
    cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    combos = list(product(
        PARAM_GRID["alpha"],
        PARAM_GRID["learning_rate"],
        PARAM_GRID["max_iter"],
    ))
    total = len(combos)
    print(f"Grid: {total} combinations × {N_FOLDS} folds = {total * N_FOLDS} fits\n")

    # Pre-transform all folds once — avoids re-fitting FeatureBuilder
    # on every hyperparameter combo (expensive and redundant).
    print("Pre-transforming folds (FeatureBuilder + FeaturePreprocessor)…")
    folds = []
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr_raw  = X_train.iloc[tr_idx].reset_index(drop=True)
        X_val_raw = X_train.iloc[val_idx].reset_index(drop=True)
        y_tr      = y_train[tr_idx]
        y_val     = y_train[val_idx]

        X_tr_proc, X_val_proc = build_features(X_tr_raw, X_val_raw)
        sw = compute_sample_weight("balanced", y_tr)
        folds.append((X_tr_proc, X_val_proc, y_tr, y_val, sw))
        print(f"  Fold {fold_idx + 1}/{N_FOLDS} done")

    print()
    records = []
    for i, (alpha, lr, max_iter) in enumerate(combos, 1):
        fold_scores = []
        for (X_tr_proc, X_val_proc, y_tr, y_val, sw) in folds:
            pipe   = build_sklearn_pipeline(alpha, lr, max_iter)
            scores = score_fold(pipe, X_tr_proc, y_tr, X_val_proc, y_val, sw)
            fold_scores.append(scores)

        pr_aucs    = [s["pr_auc"]    for s in fold_scores]
        roc_aucs   = [s["roc_auc"]   for s in fold_scores]
        macro_f1s  = [s["macro_f1"]  for s in fold_scores]
        pr_aucs_tr = [s["pr_auc_tr"] for s in fold_scores]

        records.append({
            "alpha":              alpha,
            "learning_rate":      lr,
            "max_iter":           max_iter,
            "val_pr_auc_mean":    np.mean(pr_aucs),
            "val_pr_auc_std":     np.std(pr_aucs),
            "val_roc_auc_mean":   np.mean(roc_aucs),
            "val_macro_f1_mean":  np.mean(macro_f1s),
            "train_pr_auc_mean":  np.mean(pr_aucs_tr),
            "generalisation_gap": np.mean(pr_aucs_tr) - np.mean(pr_aucs),
        })

        if i % 10 == 0 or i == total:
            best_so_far = max(r["val_pr_auc_mean"] for r in records)
            print(f"  [{i:>3}/{total}]  best PR-AUC so far: {best_so_far:.4f}")

    results_df = (
        pd.DataFrame(records)
        .sort_values("val_pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_csv = os.path.join(ARTIFACTS_DIR, "tuning_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nFull results saved → {out_csv}")
    return results_df


# ── Reporting + final fit ─────────────────────────────────────────────────────

def report_and_save(
    results_df: pd.DataFrame,
    X_train: pd.Series,
    y_train: np.ndarray,
) -> Pipeline:
    best = results_df.iloc[0]

    print("\n── Best hyperparameters ─────────────────────────────────────────")
    print(f"  alpha         = {best['alpha']}")
    print(f"  learning_rate = {best['learning_rate']}")
    print(f"  max_iter      = {int(best['max_iter'])}")

    print("\n── CV performance (best combo) ──────────────────────────────────")
    print(f"  PR-AUC (primary)  val:  {best['val_pr_auc_mean']:.4f}  ±{best['val_pr_auc_std']:.4f}")
    print(f"  ROC-AUC           val:  {best['val_roc_auc_mean']:.4f}")
    print(f"  Macro-F1          val:  {best['val_macro_f1_mean']:.4f}")
    print(f"  Train PR-AUC:          {best['train_pr_auc_mean']:.4f}")

    gap  = best["generalisation_gap"]
    flag = "⚠️  overfit — consider higher alpha" if gap > 0.08 else "✓ healthy"
    print(f"  Generalisation gap:    {gap:.4f}  {flag}")

    print("\n── Top 5 combos by PR-AUC ───────────────────────────────────────")
    print(results_df.head(5)[
        ["alpha", "learning_rate", "max_iter",
         "val_pr_auc_mean", "val_pr_auc_std", "generalisation_gap"]
    ].to_string(index=False))

    # ── Refit on full training data ───────────────────────────────────────
    print("\nRefitting best pipeline on full training set…")
    X_train_proc, _ = build_features(X_train)
    sample_weights  = compute_sample_weight("balanced", y_train)

    best_pipe = build_sklearn_pipeline(
        alpha=best["alpha"],
        learning_rate=best["learning_rate"],
        max_iter=int(best["max_iter"]),
    )
    best_pipe.fit(X_train_proc, y_train, clf__sample_weight=sample_weights)

    # ── Sparsity report ───────────────────────────────────────────────────
    clf    = best_pipe.named_steps["clf"]
    n_zero = int(np.sum(clf.coef_ == 0))
    n_feat = len(clf.coef_)
    print(f"\n── Sparsity check ───────────────────────────────────────────────")
    print(f"  Zeroed out: {n_zero}/{n_feat} coefficients  ({100*n_zero/n_feat:.1f}% sparse)")

    if n_zero == 0:
        print("  ⚠️  Nothing zeroed — alpha may be too low")
    elif n_feat - n_zero < 5:
        print("  ⚠️  Almost everything zeroed — alpha may be too aggressive")
    else:
        print("  ✓ Lasso pruned the feature space as expected")

    scaler = best_pipe.named_steps["scaler"]
    if hasattr(scaler, "feature_names_in_"):
        feat_names = scaler.feature_names_in_
        top10      = np.argsort(np.abs(clf.coef_))[::-1][:10]
        print("\n  Top-10 features by |coef|")
        print("  (EDA-4 expects vader_neg, second_person_density,")
        print("   profanity_count, uppercase_ratio near the top)")
        print(f"  {'#':<4} {'feature':<35} {'coef':>8}")
        print(f"  {'─'*4} {'─'*35} {'─'*8}")
        for rank, idx in enumerate(top10, 1):
            print(f"  {rank:<4} {feat_names[idx]:<35} {clf.coef_[idx]:>+.4f}")

    # ── Pickle for eval sprint handoff ────────────────────────────────────
    out_pkl = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"\nBest fitted pipeline → {out_pkl}")
    print("  Eval sprint usage:")
    print("    pipe    = pickle.load(open('analysis/models/artifacts/best_model.pkl', 'rb'))")
    print("    y_proba = pipe.predict_proba(X_test_proc)[:, 1]")
    print("    y_pred  = pipe.predict(X_test_proc)")
    print("    (X_test_proc must be pre-transformed via FeatureBuilder + FeaturePreprocessor)")

    return best_pipe


# ─────────────────────────────────────────────────────────────────────────────

def main(processed_path: str):
    dp = DataPipeline(processed_path=processed_path, label_columns=TARGET_COL)
    X_train_raw, _, y_train_raw, _ = dp.get_data()

    X_train = X_train_raw if isinstance(X_train_raw, pd.Series) else X_train_raw.squeeze()

    y_train = (
        y_train_raw[TARGET_COL].values
        if isinstance(y_train_raw, pd.DataFrame)
        else np.asarray(y_train_raw)
    )

    print(f"Train shape:      {X_train.shape}")
    print(f"Toxic rate:       {y_train.mean():.3%}")
    print(f"Imbalance ratio:  {(1 - y_train.mean()) / y_train.mean():.1f}:1")
    print("→ Using balanced sample weights to counter EDA-2 imbalance\n")

    results_df = tune(X_train, y_train)
    report_and_save(results_df, X_train, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_path",
        default="./data/processed/test_train_data.pkl",
        help="Path to the pre-split .pkl produced by DataPipeline",
    )
    args = parser.parse_args()
    main(args.processed_path)