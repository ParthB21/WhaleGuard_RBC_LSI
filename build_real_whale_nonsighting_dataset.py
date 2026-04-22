"""
Build a non-sighting whale dataset (Presence=0) using the same environment-fetching
pipeline as build_real_whale_dataset.py.

The script:
1) Reads visual sightings from otherData/23305_RWSAS.csv.
2) Uses the same cleaned date range as sightings.
3) Samples days with no sightings and realistic locations from the sightings pool.
4) Adds season + environmental features.
5) Aligns final columns to an existing reference CSV schema so files can be merged.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from build_real_whale_dataset import (
    EnvConfig,
    VISUAL_CSV,
    add_environment_features,
    add_season,
    load_visual_positives,
    make_run_directory,
)


PROJECT_ROOT = str(Path(__file__).resolve().parent)


def _sample_nonsighting_rows(
    visual_df: pd.DataFrame,
    n_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    if visual_df.empty:
        raise ValueError("Visual sightings data is empty; cannot sample non-sighting days.")

    visual_df = visual_df.copy()
    visual_df["SIGHTINGDATE"] = pd.to_datetime(visual_df["SIGHTINGDATE"], errors="coerce")
    visual_df = visual_df.dropna(subset=["SIGHTINGDATE", "LAT", "LON"])
    if visual_df.empty:
        raise ValueError("No valid visual rows after date/coordinate cleaning.")

    date_min = visual_df["SIGHTINGDATE"].min().normalize()
    date_max = visual_df["SIGHTINGDATE"].max().normalize()

    sighting_dates = set(visual_df["SIGHTINGDATE"].dt.normalize().tolist())
    all_days = pd.date_range(date_min, date_max, freq="D")
    nonsighting_days = pd.DatetimeIndex([d for d in all_days if d not in sighting_dates])

    if len(nonsighting_days) == 0:
        raise ValueError("No non-sighting days available within the sightings date range.")

    coords = visual_df[["LAT", "LON"]].drop_duplicates().reset_index(drop=True)
    if coords.empty:
        raise ValueError("No valid coordinates available for non-sighting sampling.")

    rng = np.random.default_rng(random_seed)
    needed = int(n_rows)

    sampled = pd.DataFrame(columns=["SIGHTINGDATE", "LAT", "LON"])
    batch_size = max(needed * 2, 10000)

    while len(sampled) < needed:
        date_idx = rng.integers(0, len(nonsighting_days), size=batch_size)
        coord_idx = rng.integers(0, len(coords), size=batch_size)

        batch = pd.DataFrame(
            {
                "SIGHTINGDATE": nonsighting_days.values[date_idx],
                "LAT": coords.iloc[coord_idx]["LAT"].to_numpy(),
                "LON": coords.iloc[coord_idx]["LON"].to_numpy(),
            }
        )

        sampled = pd.concat([sampled, batch], ignore_index=True)
        sampled = sampled.drop_duplicates(subset=["SIGHTINGDATE", "LAT", "LON"])

    sampled = sampled.head(needed).copy()
    sampled["Presence"] = 0
    sampled["SOURCE"] = "Visual"
    sampled["SOURCE_DETAIL"] = "RWSAS non-sighting days"
    sampled["PACKAGE_ID"] = pd.NA
    return sampled


def _align_to_reference_schema(df: pd.DataFrame, reference_csv: Optional[str]) -> pd.DataFrame:
    if not reference_csv or not os.path.exists(reference_csv):
        return df

    ref_cols = list(pd.read_csv(reference_csv, nrows=0).columns)
    out = df.copy()

    for col in ref_cols:
        if col not in out.columns:
            out[col] = np.nan

    out = out[ref_cols]
    return out


def build_nonsighting_dataset(
    output_csv: str,
    report_json: str,
    env_cfg: EnvConfig,
    n_rows: int,
    random_seed: int,
    reference_csv: Optional[str],
) -> Tuple[pd.DataFrame, dict]:
    visual = load_visual_positives(VISUAL_CSV)
    sampled = _sample_nonsighting_rows(visual, n_rows=n_rows, random_seed=random_seed)

    sampled = add_season(sampled)
    sampled = add_environment_features(sampled, env_cfg=env_cfg)
    sampled = sampled.sort_values(["SIGHTINGDATE", "LAT", "LON"]).reset_index(drop=True)

    sampled = _align_to_reference_schema(sampled, reference_csv=reference_csv)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sampled.to_csv(output_csv, index=False)

    report = {
        "output_csv": output_csv,
        "total_rows": int(len(sampled)),
        "positive_rows": int((sampled["Presence"] == 1).sum()) if "Presence" in sampled.columns else 0,
        "negative_rows": int((sampled["Presence"] == 0).sum()) if "Presence" in sampled.columns else int(len(sampled)),
        "requested_rows": int(n_rows),
        "date_range": {
            "min": str(pd.to_datetime(visual["SIGHTINGDATE"]).min().date()) if not visual.empty else None,
            "max": str(pd.to_datetime(visual["SIGHTINGDATE"]).max().date()) if not visual.empty else None,
        },
        "sampling": {
            "source_dates": "Days in sightings range excluding sighting days",
            "source_coords": "Unique cleaned RWSAS coordinates",
            "random_seed": int(random_seed),
        },
        "env_mode": env_cfg.mode,
        "reference_schema_csv": reference_csv,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return sampled, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-sighting whale rows (Presence=0) with the same schema as an existing sightings CSV."
    )
    parser.add_argument("--mode", choices=("coastwatch", "paper"), default="coastwatch")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--report", default=None, help="Output JSON report path.")
    parser.add_argument(
        "--reference-csv",
        default=os.path.join(PROJECT_ROOT, "data", "processed", "REAL_Whale_Training_Data_sample.csv"),
        help="Reference CSV whose header/column order will be enforced.",
    )
    parser.add_argument("--rows", type=int, default=10000, help="Number of non-sighting rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--random-run-dir", action="store_true", help="Write outputs under data/runs/run_<timestamp>_<hex>/.")

    parser.add_argument("--no-env", action="store_true", help="Skip environmental API fetching.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep between environmental requests.")
    parser.add_argument("--connect-timeout", type=int, default=5, help="HTTP connect timeout.")
    parser.add_argument("--read-timeout", type=int, default=12, help="HTTP read timeout.")
    parser.add_argument("--max-seconds-per-var", type=int, default=25, help="Time budget per variable fetch.")
    parser.add_argument("--env-cache", default=None, help="Path to persistent env lookup cache CSV.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Save env cache every N query points.")
    parser.add_argument("--max-env-points-per-run", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Less terminal output.")
    return parser.parse_args()


def _resolve_output_paths(args: argparse.Namespace) -> Tuple[str, str, Optional[str]]:
    run_dir: Optional[str] = None
    suf = "coastwatch" if args.mode == "coastwatch" else "paper"
    if args.random_run_dir:
        run_dir = make_run_directory(PROJECT_ROOT)
        output_csv = args.output or os.path.join(run_dir, f"REAL_Whale_NoSightings_{args.rows}_{suf}.csv")
        report_json = args.report or os.path.join(run_dir, f"REAL_Whale_NoSightings_{args.rows}_{suf}_report.json")
        return output_csv, report_json, run_dir

    output_csv = args.output or os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        f"REAL_Whale_NoSightings_{args.rows}_{suf}.csv",
    )
    report_json = args.report or os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        f"REAL_Whale_NoSightings_{args.rows}_{suf}_report.json",
    )
    return output_csv, report_json, run_dir


if __name__ == "__main__":
    args = parse_args()
    output_csv, report_json, run_dir = _resolve_output_paths(args)

    if args.env_cache:
        resolved_cache = args.env_cache
    elif run_dir:
        # Keep per-run cache beside the run outputs so progress is visible while collecting.
        resolved_cache = os.path.join(run_dir, f"env_lookup_cache_{args.mode}.csv")
    else:
        resolved_cache = os.path.join(PROJECT_ROOT, "data", "processed", f"env_lookup_cache_{args.mode}.csv")

    cfg = EnvConfig(
        request_timeout_connect=args.connect_timeout,
        request_timeout_read=args.read_timeout,
        polite_sleep_seconds=args.sleep,
        max_seconds_per_var=args.max_seconds_per_var,
        enable_env_fetch=not args.no_env,
        mode=args.mode,
        verbose=not args.quiet,
        env_cache_path=resolved_cache,
        checkpoint_every_points=args.checkpoint_every,
        max_env_points_per_run=args.max_env_points_per_run,
    )

    print(f"Mode: {args.mode} | rows: {args.rows} | output: {output_csv}")
    if run_dir:
        print(f"Run directory: {run_dir}")
    print(f"Env cache: {cfg.env_cache_path}")

    df, _ = build_nonsighting_dataset(
        output_csv=output_csv,
        report_json=report_json,
        env_cfg=cfg,
        n_rows=args.rows,
        random_seed=args.seed,
        reference_csv=args.reference_csv,
    )
    print(f"Saved: {output_csv}")
    print(f"Rows: {len(df)} | positives: {(df['Presence'] == 1).sum()} | negatives: {(df['Presence'] == 0).sum()}")
    print(f"Report: {report_json}")
