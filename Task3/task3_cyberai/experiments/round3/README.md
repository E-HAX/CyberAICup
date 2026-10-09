# Round-3 scratch scripts

These are the throwaway scripts that produced the round-3 findings, kept as
evidence of what was actually run. They were executed from the repository root
with paths of the form `exp/...`; those paths no longer exist, so the scripts
are **not runnable as-is** and are superseded by production code:

| Scratch script | Superseded by |
|---|---|
| `ceiling.py`, `ceiling2.py`, `permode.py`, `zoom_sub.py`, `zoom_look.py`, `err_map.py` | `src/analysis/geometry.py` |
| `hier.py`, `mix15.py`, `modehead.py`, `base_cv.py` | `src/models/hier.py`, `scripts/train.py` |
| `zoom_rule.py`, `f1_decision.py`, `one_bias.py`, `balanced.py` | measurements recorded in `ABLATION3.md` §5 |
| `final.py`, `decide.py`, `write_sub.py` | `scripts/predict.py` |
| `mirage_transfer.py` | closed branch, result in `ABLATION3.md` §6 |

Read them for the exact procedure behind a number; run the production modules
to reproduce it.
