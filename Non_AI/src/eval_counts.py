"""
src/eval_counts.py
------------------
Compare ground-truth object counts with the CV-only pipeline output.

Usage
-----
python -m src.eval_counts  --gt path/to/ground_truth.json
# or, if you want to compare against a specific summary file:
python -m src.eval_counts  --gt ground_truth.json  --pred outputs/logs/summary_20240714_101533.json
"""
from __future__ import annotations
import argparse, json, re, logging, pathlib, statistics, sys
from typing import Dict

from .config import LOG_DIR

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
def _load_json(p: pathlib.Path) -> Dict[str, int] | None:
    try:
        txt = p.read_text()
        # tolerate files without surrounding braces or with trailing commas
        txt = txt.strip()
        if not txt:
            return {}  # An empty file is not an error, but results in an empty dict.
        if not txt.startswith("{"):
            txt = "{\n" + txt + "\n}"
        txt = re.sub(r",\s*}", "}", txt)  # kill final trailing comma
        return json.loads(txt)
    except json.JSONDecodeError as e:
        log.error("Failed to parse JSON from %s: %s", p.name, e)
        return None
    except Exception as e:
        log.error("Failed to read or parse JSON from %s: %s", p, e)
        return None

def _latest_summary() -> pathlib.Path | None:
    summaries = sorted(LOG_DIR.glob("summary_*.json"))
    return summaries[-1] if summaries else None

# --------------------------------------------------------------------------- #
def main() -> None:
    """Entry point for command-line execution."""
    ap = argparse.ArgumentParser(description="Compare GT vs CV2 counts")
    ap.add_argument("--gt",  required=True, help="Ground-truth JSON file")
    ap.add_argument("--pred", help="Pipeline summary JSON (defaults to newest)")
    args = ap.parse_args()

    gt_p   = pathlib.Path(args.gt)
    pred_p = pathlib.Path(args.pred) if args.pred else None

    run_evaluation(gt_p, pred_p)

def run_evaluation(gt_path: pathlib.Path, pred_path: pathlib.Path | None) -> None:
    """
    Compares ground truth and prediction JSON files and prints a report.
    This function is designed to be importable and used by other modules.
    """
    if pred_path is None:
        pred_path = _latest_summary()

    if not pred_path or not pred_path.exists():
        log.error("No summary JSON found – run the counter first.")
        return

    gt = _load_json(gt_path)
    if gt is None:
        return # Error already logged by _load_json

    pred = _load_json(pred_path)
    if pred is None:
        return # Error already logged by _load_json

    rows, abs_errs, pct_errs = [], [], []
    for fname, true_cnt in gt.items():
        cv_cnt = pred.get(fname, 0)
        err    = cv_cnt - true_cnt
        abs_errs.append(abs(err))
        pct_err = abs(err) / true_cnt * 100 if true_cnt > 0 else 0.0
        pct_errs.append(pct_err)
        rows.append((fname, true_cnt, cv_cnt, err, pct_err))

    # ------------------------------------------------------------------ #
    print(f"\nComparison  (GT = {gt_path.name},  Pred = {pred_path.name})\n")
    print(f"{'image':35s}  {'GT':>5s}  {'CV2':>5s}  {'Δ':>5s}  {'|Δ|%':>7s}")
    print("-"*62)
    for r in sorted(rows):
        print(f"{r[0]:35s}  {r[1]:5d}  {r[2]:5d}  {r[3]:5d}  {r[4]:6.1f}%")
    print("-"*62)
    mae  = statistics.mean(abs_errs) if abs_errs else 0.0
    mape = statistics.mean(pct_errs) if pct_errs else 0.0
    print(f"MAE  = {mae:.2f}   |   MAPE = {mape:.2f}%   (n={len(rows)})\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s | %(message)s")
    main()
