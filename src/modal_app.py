"""Modal application for the RTC encrypted-traffic classification task.

Everything runs remotely: EDA, feature building, hyperparameter search, the
transformer and the auxiliary-dataset ingestion. The local machine only edits
source files and issues `modal run`.

Layout
    Volumes   rtc-work    inputs, features, out-of-fold predictions, reports
              rtc-mirage  the auxiliary MIRAGE corpus (raw and reduced)
              rtc-cache   checkpoints, Optuna journal, framework caches
    Images    IMG_BASE    numeric + plotting stack
              IMG_ML      IMG_BASE plus boosted-tree and search libraries
              IMG_GPU     IMG_ML plus torch on CUDA

Usage
    modal run src/modal_app.py::seed
    modal run src/modal_app.py::eda
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

APP_NAME = "rtc-cyberai"

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "Task3" / "RTC_CyberAICup2026"

PY_VERSION = "3.12"

# Pinned so an image rebuilt weeks from now reproduces today's numbers.
BASE_PKGS = [
    "numpy==2.2.6",
    "pandas==2.2.3",
    "scipy==1.15.2",
    "scikit-learn==1.6.1",
    "matplotlib==3.10.1",
    "seaborn==0.13.2",
    "pyarrow==19.0.1",
    "tabulate==0.9.0",
]
ML_PKGS = [
    "xgboost==2.1.4",
    "lightgbm==4.6.0",
    "catboost==1.2.7",
    "optuna==4.2.1",
    "shap==0.46.0",
    "umap-learn==0.5.7",
    "imbalanced-learn==0.13.0",
]
GPU_PKGS = ["torch==2.6.0"]

ENV = {
    "PYTHONPATH": "/root/src",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    # Parallelism lives at the Modal layer: each container stays single-minded
    # so that a wide fan-out is not fighting itself over cores.
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RTC_WORK": "/work",
    "RTC_MIRAGE": "/mirage",
    "RTC_CACHE": "/cache",
}

_base_build = (
    modal.Image.debian_slim(python_version=PY_VERSION)
    .apt_install("libgomp1", "curl")
    .uv_pip_install(*BASE_PKGS)
    .env(ENV)
)
_ml_build = _base_build.uv_pip_install(*ML_PKGS)
_gpu_build = _ml_build.uv_pip_install(*GPU_PKGS, extra_index_url="https://download.pytorch.org/whl/cu124")
_tabpfn_build = _gpu_build.uv_pip_install("tabpfn", "tabicl").env(
    {"HF_HOME": "/cache/hf", "TABPFN_MODEL_CACHE_DIR": "/cache/tabpfn"}
)


def _with_locals(image: modal.Image) -> modal.Image:
    """Attach project source and the competition CSVs.

    Local directories are mounted at runtime, so editing a module does not
    invalidate the image layers and no rebuild is triggered.
    """
    return image.add_local_dir(SRC_DIR, "/root/src").add_local_dir(
        DATA_DIR, "/root/data_seed"
    )


IMG_BASE = _with_locals(_base_build)
IMG_ML = _with_locals(_ml_build)
IMG_GPU = _with_locals(_gpu_build)
IMG_TABPFN = _with_locals(_tabpfn_build)

work_vol = modal.Volume.from_name("rtc-work", create_if_missing=True)
mirage_vol = modal.Volume.from_name("rtc-mirage", create_if_missing=True)
cache_vol = modal.Volume.from_name("rtc-cache", create_if_missing=True)

WORK_ONLY = {"/work": work_vol}
WORK_CACHE = {"/work": work_vol, "/cache": cache_vol}
ALL_VOLS = {"/work": work_vol, "/mirage": mirage_vol, "/cache": cache_vol}

app = modal.App(APP_NAME)


def _bootstrap() -> None:
    if "/root/src" not in sys.path:
        sys.path.insert(0, "/root/src")


# ------------------------------------------------------------ functions ----


@app.function(image=IMG_BASE, volumes=WORK_ONLY, timeout=600)
def seed_data() -> dict:
    """Copy the competition CSVs from the image into the work Volume."""
    import shutil

    _bootstrap()
    import config as C

    C.ensure_dirs()
    copied = {}
    for name in ("Training_set.csv", "Testing_set.csv"):
        src = C.SEED_DIR / name
        dst = C.DATA_DIR / name
        shutil.copyfile(src, dst)
        copied[name] = dst.stat().st_size
    work_vol.commit()
    return copied


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=8.0, memory=16384, timeout=3600)
def run_eda() -> dict:
    """Phase 1: the full exploratory analysis, written to the work Volume."""
    _bootstrap()
    import config as C  # noqa: F401  (sets up paths)
    from eda.run_eda import main

    result = main()
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=8.0, memory=16384, timeout=3600)
def build_features() -> dict:
    """Phase 2: cache the deterministic feature blocks, folds and selections."""
    _bootstrap()
    from features.pipeline import build_and_cache

    meta = build_and_cache()
    work_vol.commit()
    return meta


@app.function(
    image=IMG_ML,
    volumes=WORK_ONLY,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    retries=2,
    max_containers=80,
)
def fit_eval(spec: dict) -> dict:
    """The fan-out unit: score one grid point and store its out-of-fold matrix."""
    _bootstrap()
    from models.evaluate import evaluate_point

    result = evaluate_point(spec)
    work_vol.commit()
    return result


@app.function(
    image=IMG_TABPFN,
    gpu="L4",
    volumes={"/work": work_vol, "/cache": cache_vol},
    cpu=4.0,
    memory=16384,
    timeout=7200,
    retries=1,
    max_containers=6,
)
def fit_eval_gpu(spec: dict) -> dict:
    """Same fan-out contract as `fit_eval`, for families that need a GPU.

    Capped hard: each container loads foundation-model weights, so a wide fan-out
    would spend more time downloading than fitting.
    """
    _bootstrap()
    from models.evaluate import evaluate_point

    result = evaluate_point(spec)
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=3600)
def run_controls() -> dict:
    """Negative controls N1 and N2 - the round is void if these fail."""
    _bootstrap()
    from analysis.controls import run

    result = run()
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=3600)
def run_tuple_lookup() -> dict:
    """Experiment A1: is the test set answerable by tuple lookup?"""
    _bootstrap()
    from analysis.tuple_lookup import run

    result = run()
    work_vol.commit()
    return result


@app.function(image=IMG_TABPFN, gpu="L4", volumes={"/cache": cache_vol}, cpu=2.0, timeout=1800)
def inspect_model_params(model: str = "tabicl") -> dict:
    """Read a model's real constructor signature before building a grid over it."""
    _bootstrap()
    import inspect

    if model == "tabicl":
        from tabicl import TabICLClassifier as M
    else:
        from tabpfn import TabPFNClassifier as M
    sig = inspect.signature(M.__init__)
    return {
        k: str(v.default)
        for k, v in sig.parameters.items()
        if k not in ("self", "args", "kwargs")
    }


@app.function(
    image=IMG_TABPFN,
    volumes={"/cache": cache_vol},
    cpu=2.0,
    memory=8192,
    timeout=3600,
)
def fetch_tabpfn_weights(
    repo: str = "Prior-Labs/TabPFN-v2-clf", filename: str = "tabpfn-v2-classifier.ckpt"
) -> dict:
    """Cache the published TabPFN classifier checkpoint on the cache volume."""
    _bootstrap()
    import shutil
    import urllib.request
    from pathlib import Path

    dest_dir = Path("/cache/tabpfn")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if not dest.exists():
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as fh:
            shutil.copyfileobj(r, fh)
    # Keep the licence text next to the weights so the terms travel with them.
    lic = dest_dir / "LICENSE.txt"
    if not lic.exists():
        try:
            with urllib.request.urlopen(
                f"https://huggingface.co/{repo}/resolve/main/LICENSE.txt", timeout=60
            ) as r, open(lic, "wb") as fh:
                shutil.copyfileobj(r, fh)
        except Exception:
            pass
    cache_vol.commit()
    return {"path": str(dest), "bytes": dest.stat().st_size, "licence_saved": lic.exists()}


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=8.0, memory=32768, timeout=10800)
def run_call_recovery(min_auc: float = 0.75, min_purity: float = 0.90) -> dict:
    """Item 3 / D branch: same-call metric, clustering, correlated-flow aggregation."""
    _bootstrap()
    from analysis.call_recovery import run

    result = run(min_auc=min_auc, min_purity=min_purity)
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=1800)
def compare_families(model_a: str = "lgbm", model_b: str = "tabicl") -> dict:
    """Paired McNemar comparison of two families' strongest stored points."""
    _bootstrap()
    from analysis.compare import compare_best

    result = compare_best(model_a, model_b)
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=3600)
def run_lookup_override() -> dict:
    """Experiment A3: is a tuple-lookup override better than the model?"""
    _bootstrap()
    from analysis.tuple_lookup import override_gain

    result = override_gain()
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=5400)
def run_decision_layer() -> dict:
    """Experiments C1 and C2/C3 over the stored out-of-fold matrices."""
    _bootstrap()
    import json

    import numpy as np

    import config as C
    from models.calibrate import evaluate as calibrate_eval
    from models.decision import evaluate as decision_eval
    import io_utils as IO

    ens = json.loads((C.MODELS_DIR / "ensemble.json").read_text())
    best = max(ens["members"], key=lambda m: m["group_acc"])
    z = np.load(best["path"], allow_pickle=False)
    prob = z["oof_group"].astype(np.float64)
    y = IO.labels_to_ids(IO.load_train()["label"])

    c1, calibrated = calibrate_eval(prob, y)
    pick = c1["selected"]
    prob_cal = prob if pick == "raw" else calibrated[pick]

    result = {
        "member": best["name"],
        "C1_calibration": c1,
        "C1_selected": pick,
        "C2_on_raw": decision_eval(prob, y),
        "C2_on_calibrated": decision_eval(prob_cal, y),
    }
    np.save(C.MODELS_DIR / "oof_calibrated.npy", prob_cal.astype(np.float32))
    (C.REPORTS_DIR / "ablation2_decision.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=2.0, memory=8192, timeout=21600)
def tabicl_sweep(bags: list[int] | None = None, probes: bool = False) -> dict:
    """Items 1 and 2: seed-bagging, the real TabICL knobs, and the probe block.

    The model was adopted on a single untuned run, so this sweeps the three
    parameters that actually exist in its constructor and averages over seeds -
    an in-context model at n=1,285 is variance-limited, not capacity-limited.
    """
    _bootstrap()
    from models.grids import FINAL_REPEATS

    specs = []
    for bag in (bags or [1, 5]):
        for n_est in (8, 16, 32):
            for temp in (0.75, 0.9, 1.05):
                specs.append(
                    {
                        "model": "tabicl",
                        "params": {"n_estimators": n_est, "softmax_temperature": temp},
                        "features": "full",
                        "target_feats": False,
                        "probes": probes,
                        "bag": bag,
                        "repeats": FINAL_REPEATS,
                        "with_test": True,
                    }
                )
    results = [r for r in fit_eval_gpu.map(specs, order_outputs=False) if r]
    tag = "ablation2-item2-probes" if probes else "ablation2-item1-bagging"
    summary = collect_results.remote(results, tag)
    return {
        "summary": summary,
        "arms": sorted(
            [
                {
                    "bag": r["spec"].get("bag", 1),
                    "params": r["spec"]["params"],
                    "probes": r["spec"].get("probes", False),
                    "group_acc": r["group_acc"],
                    "group_sd": r["group_acc_std"],
                    "plain_acc": r.get("plain_acc"),
                    "macro_f1": r["group_macro_f1"],
                    "fit_seconds": r["fit_seconds"],
                }
                for r in results
            ],
            key=lambda d: -d["group_acc"],
        )[:12],
    }


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=2.0, memory=8192, timeout=21600)
def tabpfn_stage(
    feature_sets: list[str] | None = None, model: str = "tabicl", ckpt: str = ""
) -> dict:
    """Experiment B1/B2: tabular foundation model across feature-set widths."""
    _bootstrap()
    from models.grids import FINAL_REPEATS

    sets = feature_sets or ["full", "top120", "top60"]
    params = {"model_path": ckpt} if (ckpt and model == "tabpfn") else {}
    specs = [
        {
            "model": model,
            "params": dict(params),
            "features": fs,
            "target_feats": False,
            "probes": False,
            "repeats": FINAL_REPEATS,
            "with_test": True,
        }
        for fs in sets
    ]
    results = [r for r in fit_eval_gpu.map(specs, order_outputs=False) if r]
    summary = collect_results.remote(results, "ablation2-B1")
    return {
        "summary": summary,
        "arms": sorted(
            [
                {"features": r["features"], "group_acc": r["group_acc"],
                 "plain_acc": r.get("plain_acc"), "macro_f1": r["group_macro_f1"],
                 "fit_seconds": r["fit_seconds"]}
                for r in results
            ],
            key=lambda d: -d["group_acc"],
        ),
    }


@app.function(image=IMG_ML, volumes=WORK_CACHE, cpu=2.0, memory=4096, timeout=21600)
def search_driver(
    model: str = "lgbm", n_trials: int = 200, batch_size: int = 24, probes: bool = False
) -> dict:
    """Optuna over `fit_eval`, distributed by ask/tell batching.

    The study lives in this container and is journalled to the cache Volume;
    each asked batch is evaluated across as many containers as the fan-out cap
    allows, then told back. A killed driver resumes from the journal.
    """
    _bootstrap()
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    from models.optuna_search import JOURNAL, spec_from_trial

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    JOURNAL.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(JOURNAL / f"{model}.log")))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        storage=storage,
        study_name=f"rtc-{model}",
        load_if_exists=True,
    )

    done = 0
    while done < n_trials:
        size = min(batch_size, n_trials - done)
        trials = [study.ask() for _ in range(size)]
        specs = [spec_from_trial(model, t, probes=probes) for t in trials]
        results = list(fit_eval.map(specs, order_outputs=True))
        for trial, result in zip(trials, results):
            score = float(result["group_acc"]) if result else 0.0
            study.tell(trial, score)
        done += size
        best = study.best_trial
        print(f"{model}: {done}/{n_trials} trials, best={best.value:.4f}", flush=True)
        cache_vol.commit()

    best = study.best_trial
    return {
        "model": model,
        "n_trials": done,
        "best_value": float(best.value),
        "best_params": best.params,
    }


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=2.0, memory=8192, timeout=21600)
def search_stage(stage: str, limit: int = 0, top_k: int = 12, refine: bool = True) -> dict:
    """Run a whole search stage server-side.

    Orchestration lives in a container rather than in the local process so that
    a disconnected or interrupted client cannot take the sweep down with it -
    which is exactly what happened when the driver ran locally.
    """
    _bootstrap()
    from models.grids import FINAL_REPEATS, stage_specs

    specs = stage_specs(stage)
    if limit:
        specs = specs[:limit]
    print(f"stage '{stage}': {len(specs)} grid points", flush=True)

    results = [r for r in fit_eval.map(specs, order_outputs=False) if r]
    summary = collect_results.remote(results, f"{stage}-screen")
    print(f"screen: {summary}", flush=True)

    out = {"screen": summary, "n_points": len(specs)}
    if refine and top_k:
        best = sorted(results, key=lambda r: r.get("group_acc", 0.0), reverse=True)[:top_k]
        refined_specs = []
        for r in best:
            s = dict(r["spec"])
            s["repeats"] = FINAL_REPEATS
            s["with_test"] = True
            refined_specs.append(s)
        refined = [r for r in fit_eval.map(refined_specs, order_outputs=False) if r]
        out["final"] = collect_results.remote(refined, f"{stage}-final")
        out["top"] = [
            {
                "model": r["model"],
                "group_acc": r["group_acc"],
                "plain_acc": r.get("plain_acc"),
                "params": r["spec"]["params"],
            }
            for r in sorted(refined, key=lambda r: r.get("group_acc", 0.0), reverse=True)[:5]
        ]
    return out


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=2.0, memory=8192, timeout=21600)
def finalists_stage(top_k: int = 24, per_family: int = 4, probes: bool = False) -> dict:
    """Re-score the leaderboard's best points with the full repeat budget.

    Screening runs on one grouped repeat to stay cheap; the finalists are the
    only points that get the full budget plus a full-data refit with stored
    test probabilities, which is what the ensemble consumes.
    """
    _bootstrap()
    import json

    import pandas as pd

    import config as C
    from models.grids import FINAL_REPEATS

    df = pd.read_csv(C.REPORTS_DIR / "cv_results.csv").sort_values(
        "group_acc", ascending=False
    )
    picks = pd.concat([g.head(per_family) for _, g in df.groupby("model", sort=False)])
    picks = picks.sort_values("group_acc", ascending=False).head(top_k)

    specs = []
    for _, row in picks.iterrows():
        if str(row["model"]).startswith("tfm"):
            continue  # the transformer is trained by its own GPU function
        specs.append(
            {
                "model": row["model"],
                "params": json.loads(row["params"]),
                "features": row["features"],
                "target_feats": bool(row["target_feats"]),
                "probes": probes or bool(row.get("probes", False)),
                "repeats": FINAL_REPEATS,
                "with_test": True,
            }
        )
    print(f"re-scoring {len(specs)} finalists with test predictions", flush=True)
    results = [r for r in fit_eval.map(specs, order_outputs=False) if r]
    summary = collect_results.remote(results, "finalists")
    return {
        "n_finalists": len(results),
        "summary": summary,
        "top": sorted(
            [
                {"model": r["model"], "group_acc": r["group_acc"], "plain_acc": r.get("plain_acc")}
                for r in results
            ],
            key=lambda d: -d["group_acc"],
        )[:5],
    }


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=2.0, memory=8192, timeout=3600)
def rebuild_cv_results() -> dict:
    """Rebuild the search record from the stored out-of-fold files.

    Every scored point writes its own .npz, so the record can always be
    reconstructed even if the run that produced it was interrupted before it
    could aggregate.
    """
    _bootstrap()
    import json

    import numpy as np
    import pandas as pd

    import config as C

    rows = []
    for path in sorted(C.OOF_DIR.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        if "meta" not in z:
            continue
        meta = json.loads(str(z["meta"]))
        row = {k: v for k, v in meta.items() if k != "spec"}
        row["params"] = json.dumps(meta["spec"].get("params", {}), sort_keys=True)
        row["probes"] = bool(meta["spec"].get("probes", False))
        row["has_test"] = "test" in z
        row["stage"] = "rebuilt"
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates(subset=["hash"], keep="last")
    df = df.sort_values("group_acc", ascending=False)
    df.to_csv(C.REPORTS_DIR / "cv_results.csv", index=False)
    work_vol.commit()
    return {
        "n_points": int(len(df)),
        "best_group_acc": float(df["group_acc"].max()),
        "families": df["model"].value_counts().to_dict(),
    }


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=1800)
def collect_results(results: list[dict], stage: str) -> dict:
    """Merge fan-out results into the searchable cross-validation record."""
    _bootstrap()
    import json

    import pandas as pd

    import config as C

    C.ensure_dirs()
    rows = []
    for r in results:
        if not r:
            continue
        row = {k: v for k, v in r.items() if k != "spec"}
        row["params"] = json.dumps(r["spec"].get("params", {}), sort_keys=True)
        row["stage"] = stage
        rows.append(row)
    df = pd.DataFrame(rows)

    path = C.REPORTS_DIR / "cv_results.csv"
    if path.exists():
        prev = pd.read_csv(path)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=["hash"], keep="last")
    sort_col = "group_acc" if "group_acc" in df.columns else "plain_acc"
    df = df.sort_values(sort_col, ascending=False)
    df.to_csv(path, index=False)
    work_vol.commit()
    best = df.iloc[0].to_dict()
    return {
        "n_rows_total": int(len(df)),
        "best_model": best.get("model"),
        "best_group_acc": float(best.get("group_acc", float("nan"))),
        "best_plain_acc": float(best.get("plain_acc", float("nan"))),
        "path": str(path),
    }


@app.function(image=IMG_BASE, volumes={"/mirage": mirage_vol}, cpu=2.0, timeout=1800)
def mirage_inspect(n_names: int = 25) -> dict:
    """List the auxiliary archive and describe one member, over byte ranges."""
    _bootstrap()
    from transformer.mirage_stream import inspect

    return inspect(n_names=n_names)


@app.function(image=IMG_BASE, volumes={"/mirage": mirage_vol}, cpu=2.0, timeout=1800)
def mirage_schema() -> dict:
    """Compact reconnaissance of the auxiliary record layout."""
    _bootstrap()
    from transformer.mirage_stream import schema

    return schema()


@app.function(
    image=IMG_BASE,
    volumes={"/mirage": mirage_vol},
    cpu=2.0,
    memory=8192,
    timeout=14400,
    retries=3,
)
def mirage_fetch() -> dict:
    """Stream the 6.97 GB archive into the Volume, resumable across retries."""
    _bootstrap()
    from transformer.mirage_stream import fetch

    info = fetch()
    mirage_vol.commit()
    return info


@app.function(
    image=IMG_BASE,
    volumes={"/mirage": mirage_vol},
    cpu=2.0,
    memory=8192,
    timeout=7200,
    retries=2,
    max_containers=40,
)
def mirage_reduce_shard(job: dict) -> dict:
    """Fan-out unit: reduce one shard of archive members to compact arrays."""
    _bootstrap()
    from pathlib import Path

    from transformer.mirage_stream import RAW_DIR, reduce_members, save_shard

    zip_path = Path(job["zip_path"]) if job.get("zip_path") else None
    if zip_path is None:
        candidates = list(RAW_DIR.glob("*.zip"))
        zip_path = candidates[0] if candidates else None
    arrays = reduce_members(
        job["members"],
        zip_path=zip_path,
        max_packets=job.get("max_packets", 20),
        max_flows_per_member=job.get("max_flows_per_member", 4000),
        udp_only=job.get("udp_only", True),
    )
    if not arrays:
        return {"shard": job["shard_id"], "flows": 0}
    path = save_shard(arrays, job["shard_id"])
    mirage_vol.commit()
    return {
        "shard": job["shard_id"],
        "flows": int(len(arrays["lengths"])),
        "labels": int(len(arrays["label_names"])),
        "path": path,
    }


@app.function(
    image=IMG_GPU,
    gpu="L4",
    volumes=ALL_VOLS,
    cpu=4.0,
    memory=16384,
    timeout=14400,
)
def pretrain_encoder(cfg: dict | None = None) -> dict:
    """Self-supervised pretraining of PacketFormer on the auxiliary corpus."""
    _bootstrap()
    from transformer.pretrain import pretrain

    result = pretrain(cfg or {})
    cache_vol.commit()
    return result


@app.function(image=IMG_GPU, gpu="L4", volumes=ALL_VOLS, cpu=4.0, memory=16384, timeout=14400)
def train_transformer(
    cfg: dict | None = None, pretrained: str | None = None, tag: str = "tfm"
) -> dict:
    """Cross-validate PacketFormer and store it as an ensemble member."""
    _bootstrap()
    from transformer.finetune import run_cv

    result = run_cv(cfg=cfg, pretrained=pretrained, tag=tag)
    work_vol.commit()
    return result


@app.function(image=IMG_GPU, gpu="L4", volumes=ALL_VOLS, cpu=4.0, memory=16384, timeout=7200)
def build_probe_features(pretrained: str) -> dict:
    """Frozen-encoder linear probes P1-P5, emitted as feature block F12."""
    _bootstrap()
    from transformer.probe import build_probe_features as build

    result = build(pretrained)
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=3600)
def run_ensemble(
    min_score: float = 0.0, per_model: int = 6, top_n: int = 25, require_test: bool = True
) -> dict:
    """Combine stored out-of-fold matrices; refits nothing."""
    _bootstrap()
    from models.ensemble import build_ensemble

    result = build_ensemble(
        min_score=min_score, per_model=per_model, top_n=top_n, require_test=require_test
    )
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=3600)
def write_model_card() -> dict:
    """Assemble the model card from the artifacts already in the Volume."""
    _bootstrap()
    from reporting import build_model_card

    result = build_model_card()
    work_vol.commit()
    return result


@app.function(image=IMG_GPU, gpu="L4", volumes=ALL_VOLS, cpu=4.0, memory=16384, timeout=7200)
def evaluate_probes(pretrained: str = "/cache/pretrain/encoder.pt") -> dict:
    """Held-out quality and cross-corpus transfer of the linear probes."""
    _bootstrap()
    from transformer.probe_eval import evaluate

    result = evaluate(pretrained)
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=8.0, memory=16384, timeout=5400)
def run_zoom_analysis() -> dict:
    """t-SNE of the prediction space plus a focused study of the Zoom pair."""
    _bootstrap()
    from analysis.zoom_tsne import run

    result = run()
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=3600)
def write_shap() -> dict:
    """Global TreeSHAP attribution for the selected model."""
    _bootstrap()
    from reporting import build_shap

    result = build_shap()
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=8192, timeout=3600)
def make_submission() -> dict:
    """Write submission.csv from the winning blend's stored test probabilities."""
    _bootstrap()
    from predict import write_submission

    info = write_submission()
    work_vol.commit()
    return info


# ------------------------------------------------------- local entrypoints ----


@app.local_entrypoint()
def seed():
    print(seed_data.remote())


@app.local_entrypoint()
def eda():
    result = run_eda.remote()
    for k, v in result.items():
        print(f"{k}: {v}")
    print("\nFetch with: modal volume get rtc-work reports ./reports")


@app.local_entrypoint()
def features():
    meta = build_features.remote()
    print(f"base features: {meta['n_base_features']}")
    print(f"folds: {meta['folds']}   pseudo-groups: {meta['n_pseudo_groups']}")


@app.local_entrypoint()
def search(stage: str = "smoke", limit: int = 0, top_k: int = 12, refine: bool = True):
    """Distributed grid search: one container per grid point.

    The screening pass uses a single repeat per scheme; the best `top_k` points
    are then re-scored with the full repeat budget so the decisive comparison
    is made on a low-variance estimate.
    """
    _bootstrap()
    sys.path.insert(0, str(SRC_DIR))
    from models.grids import FINAL_REPEATS, stage_specs

    specs = stage_specs(stage)
    if limit:
        specs = specs[:limit]
    print(f"stage '{stage}': {len(specs)} grid points")

    results = [r for r in fit_eval.map(specs, order_outputs=False) if r]
    summary = collect_results.remote(results, f"{stage}-screen")
    print(f"screen: {summary}")

    if refine and top_k:
        key = "group_acc"
        best = sorted(results, key=lambda r: r.get(key, 0.0), reverse=True)[:top_k]
        refined_specs = []
        for r in best:
            s = dict(r["spec"])
            s["repeats"] = FINAL_REPEATS
            refined_specs.append(s)
        refined = [r for r in fit_eval.map(refined_specs, order_outputs=False) if r]
        summary = collect_results.remote(refined, f"{stage}-final")
        print(f"final: {summary}")
        for r in sorted(refined, key=lambda r: r.get(key, 0.0), reverse=True)[:5]:
            print(
                f"  {r['model']:6s} group={r['group_acc']:.4f}+-{r['group_acc_std']:.3f} "
                f"plain={r['plain_acc']:.4f} f1={r['group_macro_f1']:.4f} "
                f"{r['features']} taf={r['target_feats']} {r['spec']['params']}"
            )


@app.local_entrypoint()
def launch(stage: str = "all_trees", limit: int = 0, top_k: int = 12):
    """Start a search stage server-side and return immediately.

    The call keeps running on Modal after this process exits, so an interrupted
    terminal cannot cancel a sweep that has been running for half an hour.
    """
    call = search_stage.spawn(stage, limit, top_k, True)
    print(f"spawned search stage '{stage}' as {call.object_id}")
    print("progress: modal app logs rtc-cyberai   |   record: reports/cv_results.csv")


@app.local_entrypoint()
def launch_transformer(pretrained: str = "", tag: str = "tfm"):
    call = train_transformer.spawn(None, pretrained or None, tag)
    print(f"spawned transformer '{tag}' as {call.object_id}")


@app.local_entrypoint()
def launch_optuna(model: str = "lgbm", trials: int = 200, batch: int = 24, probes: bool = False):
    call = search_driver.spawn(model, trials, batch, probes)
    print(f"spawned Optuna study for {model} as {call.object_id}")


@app.local_entrypoint()
def launch_finalists(top_k: int = 24, per_family: int = 4, probes: bool = False):
    call = finalists_stage.spawn(top_k, per_family, probes)
    print(f"spawned finalists as {call.object_id}")


@app.local_entrypoint()
def rebuild():
    """Reconstruct the search record from the stored out-of-fold files."""
    print(rebuild_cv_results.remote())


@app.local_entrypoint()
def optuna_search(model: str = "lgbm", trials: int = 200, batch: int = 24, probes: bool = False):
    result = search_driver.remote(
        model=model, n_trials=trials, batch_size=batch, probes=probes
    )
    print(f"{result['model']}: best group accuracy {result['best_value']:.4f}")
    print(result["best_params"])


@app.local_entrypoint()
def finalists(top_k: int = 20):
    """Re-score the leaderboard's best points with test predictions attached."""
    _bootstrap()
    sys.path.insert(0, str(SRC_DIR))
    import json

    from models.grids import FINAL_REPEATS

    picks = best_points.remote(top_k)
    specs = []
    for row in picks:
        specs.append(
            {
                "model": row["model"],
                "params": json.loads(row["params"]),
                "features": row["features"],
                "target_feats": bool(row["target_feats"]),
                "repeats": FINAL_REPEATS,
                "with_test": True,
            }
        )
    print(f"re-scoring {len(specs)} finalists with test predictions")
    results = [r for r in fit_eval.map(specs, order_outputs=False) if r]
    print(collect_results.remote(results, "finalists"))


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=1.0, timeout=600)
def best_points(top_k: int = 20) -> list[dict]:
    """Best scored points so far, spread across model families for diversity."""
    _bootstrap()
    import pandas as pd

    import config as C

    df = pd.read_csv(C.REPORTS_DIR / "cv_results.csv").sort_values(
        "group_acc", ascending=False
    )
    per_family = max(2, top_k // max(1, df["model"].nunique()))
    picks = pd.concat(
        [g.head(per_family) for _, g in df.groupby("model", sort=False)]
    ).sort_values("group_acc", ascending=False)
    cols = ["model", "params", "features", "target_feats", "group_acc"]
    return picks[cols].head(top_k).to_dict("records")


@app.local_entrypoint()
def ensemble(min_score: float = 0.0, per_model: int = 6, top_n: int = 25):
    result = run_ensemble.remote(min_score=min_score, per_model=per_model, top_n=top_n)
    print(f"winner: {result['winner']}  acc={result['winner_acc']:.4f}")
    for k in ("single_best", "hill_climb", "rank_average", "stacking"):
        print(f"  {k}: {result[k]}")


@app.local_entrypoint()
def mirage_probe(out: str = "reports_dl/mirage_schema.json"):
    """Cheap schema reconnaissance on the auxiliary archive, no full download."""
    import json
    from pathlib import Path

    info = mirage_schema.remote()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(info, indent=2, default=str))
    print(f"wrote {out}")
    print(json.dumps({k: info[k] for k in ("n_members", "n_json", "sample_member",
                                           "n_flows_in_member", "flow_sections")
                      if k in info}, indent=2))


@app.local_entrypoint()
def aux_ingest(
    shards: int = 24,
    max_packets: int = 20,
    per_member: int = 4000,
    udp_only: bool = True,
    skip_download: bool = False,
):
    """Download the archive once, then reduce it across parallel containers."""
    if not skip_download:
        info = mirage_fetch.remote()
        print(f"archive: {info}")
    members = mirage_members.remote()
    print(f"{len(members)} JSON members; fanning out over {shards} shards")
    jobs = []
    for i in range(shards):
        chunk = members[i::shards]
        if chunk:
            jobs.append(
                {
                    "shard_id": i,
                    "members": chunk,
                    "max_packets": max_packets,
                    "max_flows_per_member": per_member,
                    "udp_only": udp_only,
                }
            )
    results = list(mirage_reduce_shard.map(jobs, order_outputs=False))
    total = sum(r.get("flows", 0) for r in results)
    print(f"reduced {total} flows into {len(results)} shards")


@app.function(image=IMG_BASE, volumes={"/mirage": mirage_vol}, cpu=2.0, timeout=3600)
def mirage_members() -> list[str]:
    _bootstrap()
    import zipfile

    from transformer.mirage_stream import RAW_DIR

    zip_path = sorted(RAW_DIR.glob("*.zip"))[0]
    with zipfile.ZipFile(zip_path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(".json")]


@app.local_entrypoint()
def pretrain(steps: int = 8000):
    result = pretrain_encoder.remote({"steps": steps})
    print(result)


@app.local_entrypoint()
def transformer(pretrained: str = "", tag: str = "tfm"):
    result = train_transformer.remote(pretrained=pretrained or None, tag=tag)
    print(
        f"{tag}: group={result['group_acc']:.4f}+-{result['group_acc_std']:.3f} "
        f"plain={result['plain_acc']:.4f} f1={result['group_macro_f1']:.4f} "
        f"({result['fit_seconds']:.0f}s)"
    )


@app.local_entrypoint()
def probes(pretrained: str):
    print(build_probe_features.remote(pretrained))


@app.local_entrypoint()
def model_card():
    print(write_model_card.remote())


@app.local_entrypoint()
def submit():
    info = make_submission.remote()
    print(f"rows={info['rows']} distinct_labels={info['distinct_labels']}")
    print(f"mean confidence: {info['mean_confidence']:.3f}")
    print("fetch: modal volume get rtc-work submission.csv ./submission.csv")


@app.function(
    image=IMG_TABPFN,
    gpu="L4",
    volumes={"/work": work_vol, "/cache": cache_vol},
    cpu=4.0,
    memory=16384,
    timeout=10800,
    retries=1,
    max_containers=6,
)
def hier_point(spec: dict) -> dict:
    """Score one hierarchical (application x mode) configuration.

    Runs the flat head, the application head and the five per-application mode
    heads over the stored grouped folds, then reports the flat model, the
    hierarchical composition, their average, and each of those with the derived
    Zoom rule applied. Out-of-fold and test probability matrices are stored so
    the ensemble stage never refits.
    """
    _bootstrap()
    import json
    import time

    import numpy as np

    import config as C
    from features.pipeline import Cache
    from models import hier

    cache = Cache.get()
    y = cache.y
    X = cache.base_train
    model = spec["model"]
    params = spec.get("params", {})
    mode_params = spec.get("mode_params") or params
    # The grouped scheme is stored with two repeats; asking for more would leave
    # empty probability slabs and a meaningless spread across repeats.
    available = len(cache.folds["group"]) // C.N_SPLITS
    n_rep = min(int(spec.get("repeats", 2)), available)
    splits = cache.folds["group"][: n_rep * C.N_SPLITS]

    t0 = time.time()
    audio = hier.audio_only(cache.train)
    flat = np.zeros((n_rep, len(y), len(C.LABELS)), dtype=np.float32)
    hi = np.zeros_like(flat)
    mx = np.zeros_like(flat)
    for i, (tr_idx, va_idx) in enumerate(splits):
        rep = i // C.N_SPLITS
        f, h = hier.heads_predict(
            model, params, X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], mode_params
        )
        flat[rep, va_idx] = f
        hi[rep, va_idx] = h
        mx[rep, va_idx] = hier.mixture_predict(
            model, params, X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], audio[tr_idx]
        )

    variants = {}
    matrices = {
        "flat": flat.mean(axis=0),
        "hier": hi.mean(axis=0),
        "mix": mx.mean(axis=0),
        "avg": 0.5 * flat.mean(axis=0) + 0.5 * hi.mean(axis=0),
        "avg3": (flat.mean(axis=0) + hi.mean(axis=0) + mx.mean(axis=0)) / 3.0,
    }
    for name, P in matrices.items():
        variants[name] = hier.score(y, P.argmax(axis=1))
        variants[name + "+zoom"] = hier.score(y, hier.apply_zoom_rule(P, audio))

    # Per-repeat spread on the headline variant, for the noise floor.
    per_rep = [
        hier.score(
            y, hier.apply_zoom_rule((flat[r] + hi[r] + mx[r]) / 3.0, audio)
        )["macro_f1"]
        for r in range(n_rep)
    ]

    test_arrays = {}
    if spec.get("with_test", False):
        f_te, h_te = hier.heads_predict(
            model, params, X, y, cache.base_test, mode_params
        )
        m_te = hier.mixture_predict(model, params, X, y, cache.base_test, audio)
        test_arrays = {"test_flat": f_te.astype(np.float32),
                       "test_hier": h_te.astype(np.float32),
                       "test_mix": m_te.astype(np.float32)}

    result = {
        "spec": spec,
        "variants": variants,
        "macro_f1_per_repeat": per_rep,
        "macro_f1_sd": float(np.std(per_rep)),
        "fit_seconds": round(time.time() - t0, 2),
    }
    tag = f"{model}_{abs(hash(json.dumps(spec, sort_keys=True, default=str))) % (10**10)}"
    path = C.OOF_DIR / f"hier_{tag}.npz"
    C.OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=json.dumps(result, default=str),
        oof_flat=matrices["flat"],
        oof_hier=matrices["hier"],
        oof_mix=matrices["mix"],
        **test_arrays,
    )
    result["oof_path"] = str(path)
    work_vol.commit()
    return result


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=2.0, memory=8192, timeout=21600)
def hier_stage(models: list[str] | None = None, repeats: int = 3) -> dict:
    """Run the hierarchical design across base learners and report every variant."""
    _bootstrap()

    specs = []
    for m in models or ["tabicl", "lgbm", "xgb", "rf"]:
        params = {
            "tabicl": {"n_estimators": 16},
            "lgbm": {"n_estimators": 600, "num_leaves": 31, "learning_rate": 0.05,
                     "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1},
            "xgb": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.06,
                    "colsample_bytree": 0.5, "subsample": 0.9},
            "rf": {"n_estimators": 1200, "max_features": "sqrt"},
        }.get(m, {})
        specs.append({"model": m, "params": params, "repeats": repeats,
                      "with_test": True})
    results = [r for r in hier_point.map(specs, order_outputs=False) if r]
    rows = []
    for r in results:
        for name, s in r["variants"].items():
            rows.append({"model": r["spec"]["model"], "variant": name,
                         "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                         "macro_f1_sd": r["macro_f1_sd"]})
    return {"rows": sorted(rows, key=lambda d: -d["macro_f1"]),
            "paths": [r["oof_path"] for r in results]}


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=4.0, memory=16384, timeout=3600)
def hier_submit(paths: list[str], write: bool = False) -> dict:
    """Combine the stored hierarchical runs and, optionally, write the submission.

    Selection is deliberately weak: the reported headline is the plain average
    over every base learner and every view, because picking the best of thirty
    out-of-fold numbers on 1,285 rows buys about as much selection bias as it
    buys accuracy. The Zoom rule is applied on top and is not fitted.
    """
    _bootstrap()
    import json

    import numpy as np
    import pandas as pd

    import config as C
    from features.pipeline import Cache
    from models import hier

    cache = Cache.get()
    y = cache.y
    audio_tr = hier.audio_only(cache.train)
    audio_te = hier.audio_only(cache.test)

    oof, test, meta = {}, {}, []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        name = m["spec"]["model"]
        meta.append({"model": name, "variants": m["variants"]})
        views = [k[4:] for k in d.files if k.startswith("oof_")]
        for v in views:
            oof.setdefault(v, []).append(d[f"oof_{v}"])
            if f"test_{v}" in d.files:
                test.setdefault(v, []).append(d[f"test_{v}"])

    def stack(store):
        per_view = {v: np.mean(a, axis=0) for v, a in store.items()}
        per_view["all"] = np.mean(list(per_view.values()), axis=0)
        return per_view

    O, T = stack(oof), stack(test)
    report = {}
    for v, P in O.items():
        report[v] = hier.score(y, P.argmax(axis=1))
        report[v + "+zoom"] = hier.score(y, hier.apply_zoom_rule(P, audio_tr))

    out = {"per_model": meta, "ensemble": report}
    if write and "all" in T:
        from predict import write_submission

        pred = hier.apply_zoom_rule(T["all"], audio_te)
        labels = np.asarray([C.ID_TO_LABEL[int(i)] for i in pred])
        out["submission"] = write_submission(T["all"], labels=labels)
        work_vol.commit()
    return out


@app.function(image=IMG_ML, volumes=WORK_ONLY, cpu=8.0, memory=16384, timeout=3600)
def run_geometry() -> dict:
    """Reproduce the label-geometry measurements behind the round-3 ceiling."""
    _bootstrap()
    from analysis import geometry

    return geometry.report()
