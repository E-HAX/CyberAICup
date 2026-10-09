"""Write and validate the competition submission.

The submitted file must contain exactly 327 rows in the order of
Testing_set.csv, with no header, a 1-based index in the first column and one of
the ten case-sensitive label strings in the second. Validation runs before the
file is written; a failure aborts rather than emitting a malformed file.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
import io_utils as IO

SUBMISSION = C.WORK / "submission.csv"
TEST_PROBA = C.MODELS_DIR / "test_proba.npy"


def validate(df: pd.DataFrame) -> dict:
    problems = []
    if len(df) != C.N_TEST_ROWS:
        problems.append(f"row count {len(df)} != {C.N_TEST_ROWS}")
    if df.shape[1] != 2:
        problems.append(f"column count {df.shape[1]} != 2")
    idx = df.iloc[:, 0].to_numpy()
    if not np.array_equal(idx, np.arange(1, len(df) + 1)):
        problems.append("first column is not 1..N in order")
    bad = sorted(set(df.iloc[:, 1]) - set(C.LABELS))
    if bad:
        problems.append(f"labels outside the vocabulary: {bad}")
    if problems:
        raise ValueError("submission validation failed: " + "; ".join(problems))
    return {"rows": len(df), "distinct_labels": int(df.iloc[:, 1].nunique())}


def distribution_check(pred: pd.Series) -> pd.DataFrame:
    """Compare the predicted class mix with the flows-per-call prior.

    The training flow counts divided by the 40 source calls per class give the
    number of UDP flows an application opens per call; scaled to the 10 held-out
    test calls per class it yields an expected test mix. The task description
    warns that the test split is not guaranteed uniform, so this is a sanity
    check rather than a constraint.
    """
    train_counts = IO.load_train()["label"].value_counts().reindex(C.LABELS)
    flows_per_call = train_counts / C.CALLS_PER_CLASS_TRAIN
    implied = flows_per_call * C.CALLS_PER_CLASS_TEST
    implied_share = implied / implied.sum()
    got = pred.value_counts().reindex(C.LABELS).fillna(0)
    return pd.DataFrame(
        {
            "predicted_flows": got.astype(int),
            "predicted_share": (got / got.sum()).round(4),
            "prior_share": implied_share.round(4),
            "ratio": (got / got.sum() / implied_share).round(3),
        }
    )


def write_submission(
    proba: np.ndarray | None = None, labels: np.ndarray | None = None
) -> dict:
    """`labels` lets a decision layer override the plain argmax.

    The Zoom rule in `models.hier` is a label decision rather than a change of
    belief, so it cannot be expressed by editing the probability matrix without
    misrepresenting the model's confidence.
    """
    if proba is None:
        proba = np.load(TEST_PROBA)
    test = IO.load_test()
    if len(proba) != len(test):
        raise ValueError(f"probability rows {len(proba)} != test rows {len(test)}")

    labels = IO.ids_to_labels(proba.argmax(axis=1)) if labels is None else labels
    sub = pd.DataFrame({"index": np.arange(1, len(labels) + 1), "label": labels})
    info = validate(sub)

    dist = distribution_check(sub["label"])
    dist.to_csv(C.REPORTS_DIR / "submission_distribution.csv")

    sub.to_csv(SUBMISSION, header=False, index=False)
    info["path"] = str(SUBMISSION)
    info["mean_confidence"] = float(proba.max(axis=1).mean())
    info["distribution"] = json.loads(dist.to_json())
    return info


if __name__ == "__main__":
    print(json.dumps(write_submission(), indent=2)[:2000])
