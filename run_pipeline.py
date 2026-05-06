#!/usr/bin/env python3
"""

this is the main script to run the full pipeline 

run_pipeline.py  -  master Pipeline Runner

executtes all six phases of the Grand Slam Upset Predictor pipeline in the correct order with progress logging and error handling.

Usage:
    python run_pipeline.py                # Full pipeline (Phases 1–6)
    python run_pipeline.py --from 5       # Start from Phase 5 (modelling)
    python run_pipeline.py --skip-scrape  # Skip web scraping (Phase 2)
    python run_pipeline.py --demo-nlp     # Run NLP in demo mode (no DB needed)
    python run_pipeline.py --no-cluster   # Skip clustering (Phase 6)
    python run_pipeline.py --llm          # Enable LLM zero-shot in NLP (Phase 3)

Pipeline order:
    Phase 1 — data_ingestion.py    : Download ATP+WTA data → SQLite
    Phase 2 — scraping.py          : Scrape press conference transcripts
    Phase 3 — nlp.py               : NLP processing (sentiment + fatigue)
    Phase 4 — features.py          : Feature engineering (CTFI + NLP merge)
    Phase 5 — model.py             : Model training + evaluation + plots
    Phase 6 — clustering.py        : Unsupervised player archetype analysis
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ANSI colour helpers 

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("32;1", t)
YELLOW = lambda t: _c("33;1", t)
RED    = lambda t: _c("31;1", t)
CYAN   = lambda t: _c("36;1", t)
BOLD   = lambda t: _c("1", t)


#phase definitions 

PHASES = [
    {
        "id":      1,
        "name":    "Data Ingestion (ATP + WTA → SQLite)",
        "script":  "data_ingestion.py",
        "args":    [],
        "skip_flag": None,
        "produces": ["tennis_upsets.db"],
    },
    {
        "id":      2,
        "name":    "Transcript Scraping (ASAP Sports)",
        "script":  "scraping.py",
        "args":    [],
        "skip_flag": "skip_scrape",
        "produces": [],  # writes into tennis_upsets.db
    },
    {
        "id":      3,
        "name":    "NLP Processing (sentiment + fatigue + LLM)",
        "script":  "nlp.py",
        "args":    [],   # modified below based on -llm /-demo-nlp
        "skip_flag": None,
        "produces": ["nlp_features.csv"],
    },
    {
        "id":      4,
        "name":    "Feature Engineering (CTFI + NLP merge)",
        "script":  "features.py",
        "args":    [],
        "skip_flag": None,
        "produces": ["features.csv"],
    },
    {
        "id":      5,
        "name":    "Model Training + Evaluation + Ablation",
        "script":  "model.py",
        "args":    [],
        "skip_flag": None,
        "produces": [
            "upset_model.pkl",
            "roc_curves_all.png",
            "precision_recall_curves.png",
            "confusion_matrices.png",
            "calibration_curves.png",
            "correlation_heatmap.png",
            "ctfi_upset_by_surface.png",
            "shap_importance.png",
            "metric_summary.png",
        ],
    },
    {
        "id":      6,
        "name":    "Unsupervised Clustering (Player Archetypes)",
        "script":  "clustering.py",
        "args":    [],
        "skip_flag": "no_cluster",
        "produces": [
            "player_clusters.csv",
            "cluster_pca_scatter.png",
            "cluster_upset_rates.png",
            "silhouette_curve.png",
        ],
    },
]


# runner 

def run_phase(phase: dict, python: str = sys.executable) -> bool:
    """
    Run a single pipeline phase. Returns True on success, False on failure.
    """
    script = Path(phase["script"])
    if not script.exists():
        print(RED(f"  [ERROR] {script} not found — skipping."))
        return False

    cmd = [python, str(script)] + phase["args"]
    print(CYAN(f"\n  $ {' '.join(cmd)}"))

    t0     = time.time()
    env = {**__import__("os").environ, "PYTHONWARNINGS": "ignore::UserWarning"}
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(GREEN(f"  ✓  Phase {phase['id']} completed in {elapsed:.1f}s"))
        # check that declared outputs were produced
        missing = [p for p in phase["produces"] if not Path(p).exists()]
        if missing:
            print(YELLOW(f"  [WARN] Expected output(s) not found: {missing}"))
        return True
    else:
        print(RED(f"  ✗  Phase {phase['id']} FAILED (exit code {result.returncode})"))
        return False


def banner(text: str) -> None:
    width = 62
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master pipeline runner for the Grand Slam Upset Predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from", dest="start_phase", type=int, default=1, metavar="N",
        help="Start from phase N (1–6). Default: 1."
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip Phase 2 (web scraping). Use if transcripts are already in DB."
    )
    parser.add_argument(
        "--demo-nlp", action="store_true",
        help="Run NLP (Phase 3) in demo mode on synthetic transcripts."
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable LLM zero-shot fatigue classification in Phase 3."
    )
    parser.add_argument(
        "--no-cluster", action="store_true",
        help="Skip Phase 6 (unsupervised clustering)."
    )
    args = parser.parse_args()

    # patch phase args based on CLI fla
    for phase in PHASES:
        if phase["id"] == 3:
            if args.demo_nlp:
                phase["args"].append("--demo")
            if args.llm:
                phase["args"].append("--llm")

    # build skip list 
    skip_flags = set()
    if args.skip_scrape:
        skip_flags.add("skip_scrape")
    if args.no_cluster:
        skip_flags.add("no_cluster")

    # header 
    banner("Grand Slam Upset Predictor — Full Pipeline")
    print(f"  Start time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Starting from: Phase {args.start_phase}")
    print(f"  Skipping     : {skip_flags or 'none'}")

    # execute phases 
    results: list[dict] = []
    pipeline_t0 = time.time()

    for phase in PHASES:
        if phase["id"] < args.start_phase:
            continue

        if phase["skip_flag"] and phase["skip_flag"] in skip_flags:
            print(YELLOW(f"\n[--] Phase {phase['id']}: {phase['name']} — SKIPPED"))
            results.append({"phase": phase["id"], "status": "skipped"})
            continue

        banner(f"Phase {phase['id']} / 6 — {phase['name']}")
        success = run_phase(phase)
        results.append({
            "phase":  phase["id"],
            "status": "ok" if success else "failed",
        })

        if not success:
            print(RED(f"\n[ABORT] Phase {phase['id']} failed. "
                      f"Fix the error and re-run with --from {phase['id']}."))
            break

    # summary 
    total_elapsed = time.time() - pipeline_t0
    banner("Pipeline Summary")
    status_sym = {
        "ok":      GREEN("✓"),
        "failed":  RED("✗"),
        "skipped": YELLOW("–"),
    }
    for r in results:
        phase = next(p for p in PHASES if p["id"] == r["phase"])
        sym   = status_sym[r["status"]]
        print(f"  {sym}  Phase {r['phase']}: {phase['name']}")

    n_ok     = sum(1 for r in results if r["status"] == "ok")
    n_fail   = sum(1 for r in results if r["status"] == "failed")
    n_skip   = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n  {n_ok} succeeded · {n_skip} skipped · {n_fail} failed")
    print(f"  Total elapsed: {total_elapsed/60:.1f} minutes")
    print("═" * 62)

    if n_fail == 0:
        print(GREEN("\n  All phases complete. Run:"))
        print(GREEN("    streamlit run app.py"))
        print(GREEN("  to launch the interactive dashboard.\n"))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()