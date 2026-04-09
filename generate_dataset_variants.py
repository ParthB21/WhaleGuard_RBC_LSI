"""
Generate multiple cleaned dataset variants from a source whale CSV.

Purpose:
- produce a portfolio of training-ready datasets for discussion/review
- keep original source file untouched
- write a manifest with row counts and class balance

Example:
  python generate_dataset_variants.py \
    --input data/runs/run_20260331_190615_05b7b819/whale_dataset_paper.csv \
    --output-dir data/runs/run_20260331_190615_05b7b819/dataset_variants
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Variant:
    name: str
    include_visual: bool
    min_year: int | None = None
    max_year: int | None = None


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    required = {"SIGHTINGDATE", "LAT", "LON", "Presence"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    out["SIGHTINGDATE"] = pd.to_datetime(out["SIGHTINGDATE"], errors="coerce")
    out["LAT"] = pd.to_numeric(out["LAT"], errors="coerce")
    out["LON"] = pd.to_numeric(out["LON"], errors="coerce")
    out["Presence"] = pd.to_numeric(out["Presence"], errors="coerce")

    out = out.dropna(subset=["SIGHTINGDATE", "LAT", "LON", "Presence"]).copy()
    out = out[out["LAT"].between(-90, 90) & out["LON"].between(-180, 180)].copy()
    out["Presence"] = out["Presence"].clip(0, 1).astype(int)
    out = out.drop_duplicates(subset=["SIGHTINGDATE", "LAT", "LON", "Presence"]).copy()
    return out


def _filter_source(df: pd.DataFrame, include_visual: bool) -> pd.DataFrame:
    if include_visual or "SOURCE" not in df.columns:
        return df.copy()

    src = df["SOURCE"].astype(str)
    keep = src.str.contains("Acoustic", case=False, na=False) & ~src.str.contains("Visual", case=False, na=False)
    return df[keep].copy()


def _filter_years(df: pd.DataFrame, min_year: int | None, max_year: int | None) -> pd.DataFrame:
    out = df.copy()
    if min_year is not None:
        out = out[out["SIGHTINGDATE"].dt.year >= min_year].copy()
    if max_year is not None:
        out = out[out["SIGHTINGDATE"].dt.year <= max_year].copy()
    return out


def _ensure_flags(df: pd.DataFrame, env_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c, fcol in [
        ("SST_Celsius", "SST_is_observed"),
        ("Frontal_Value", "Frontal_is_observed"),
        ("Chlorophyll_mg_m3", "Chlorophyll_is_observed"),
        ("Salinity_PSU", "Salinity_is_observed"),
        ("Water_Mass_M_WK", "Water_Mass_is_observed"),
    ]:
        if c not in out.columns:
            out[c] = np.nan
        if fcol not in out.columns:
            out[fcol] = pd.to_numeric(out[c], errors="coerce").notna().astype(int)

    if "Any_env_missing" not in out.columns:
        req = ["SST_is_observed", "Frontal_is_observed", "Chlorophyll_is_observed", "Salinity_is_observed"]
        all_obs = np.logical_and.reduce([(out[c] == 1).to_numpy() for c in req])
        out["Any_env_missing"] = (~all_obs).astype(int)
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["SIGHTINGDATE"]
    out["Year"] = dt.dt.year
    out["Month"] = dt.dt.month
    out["DayOfYear"] = dt.dt.dayofyear
    out["Month_sin"] = np.sin(2 * np.pi * out["Month"] / 12.0)
    out["Month_cos"] = np.cos(2 * np.pi * out["Month"] / 12.0)
    return out


def _median_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    medians: Dict[str, float] = {}
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        m = out[c].median()
        medians[c] = float(m) if pd.notna(m) else 0.0
        out[c] = out[c].fillna(medians[c])
    out.attrs["medians"] = medians
    return out


def _prepare_variant(input_csv: str, out_dir: str, include_visual: bool, min_year: int | None, max_year: int | None) -> Dict[str, str]:
    raw = _read_csv(input_csv)
    if raw.empty:
        raise ValueError(f"Could not read input CSV: {input_csv}")

    env_cols = ["SST_Celsius", "Frontal_Value", "Chlorophyll_mg_m3", "Salinity_PSU", "Water_Mass_M_WK"]

    clean = _basic_clean(raw)
    scoped = _filter_source(clean, include_visual=include_visual)
    scoped = _filter_years(scoped, min_year=min_year, max_year=max_year)
    scoped = _ensure_flags(scoped, env_cols=env_cols)
    scoped = _add_time_features(scoped)

    preferred = [
        "SIGHTINGDATE", "LAT", "LON", "Presence", "Season", "SOURCE", "SOURCE_DETAIL", "PACKAGE_ID",
        *env_cols,
        "SST_is_observed", "Frontal_is_observed", "Chlorophyll_is_observed", "Salinity_is_observed",
        "Water_Mass_is_observed", "Any_env_missing", "Year", "Month", "DayOfYear", "Month_sin", "Month_cos",
    ]
    cols = [c for c in preferred if c in scoped.columns]
    out_raw = scoped[cols].sort_values(["SIGHTINGDATE", "LAT", "LON"]).reset_index(drop=True)
    out_imp = _median_impute(out_raw, [c for c in env_cols if c in out_raw.columns])

    prefix = "mixed" if include_visual else "acoustic"
    raw_path = os.path.join(out_dir, f"training_ready_{prefix}_raw.csv")
    imp_path = os.path.join(out_dir, f"training_ready_{prefix}_imputed.csv")
    summary_path = os.path.join(out_dir, f"training_ready_{prefix}_summary.json")

    out_raw.to_csv(raw_path, index=False)
    out_imp.to_csv(imp_path, index=False)

    summary = {
        "input_csv": input_csv,
        "output_raw": raw_path,
        "output_imputed": imp_path,
        "include_visual": include_visual,
        "min_year": min_year,
        "max_year": max_year,
        "rows_after_scope": int(len(out_raw)),
        "presence_counts": {str(k): int(v) for k, v in out_imp["Presence"].value_counts().to_dict().items()},
        "positive_rate": float((out_imp["Presence"] == 1).mean()) if len(out_imp) else None,
        "imputation_medians": out_imp.attrs.get("medians", {}),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"raw": raw_path, "imputed": imp_path, "summary": summary_path}


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _balance_undersample(df: pd.DataFrame, ratio_neg_to_pos: float, seed: int) -> pd.DataFrame:
    if "Presence" not in df.columns:
        return df.copy()

    out = df.copy()
    y = pd.to_numeric(out["Presence"], errors="coerce").fillna(0).astype(int)
    pos = out[y == 1]
    neg = out[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return out

    rng = np.random.default_rng(seed)
    # Target relation: n_neg ~= ratio_neg_to_pos * n_pos
    # Under-sample whichever class is needed to meet that target.
    if len(neg) >= ratio_neg_to_pos * len(pos):
        # Too many negatives; keep all positives and downsample negatives.
        n_pos = len(pos)
        n_neg = int(round(ratio_neg_to_pos * n_pos))
    else:
        # Too many positives for the available negatives; keep all negatives and downsample positives.
        n_neg = len(neg)
        n_pos = int(round(n_neg / ratio_neg_to_pos)) if ratio_neg_to_pos > 0 else len(pos)

    n_pos = max(1, min(len(pos), n_pos))
    n_neg = max(1, min(len(neg), n_neg))

    sampled_pos_idx = rng.choice(pos.index.to_numpy(), size=n_pos, replace=False)
    sampled_neg_idx = rng.choice(neg.index.to_numpy(), size=n_neg, replace=False)

    sampled = pd.concat([pos.loc[sampled_pos_idx], neg.loc[sampled_neg_idx]], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def _stats(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {
            "rows": 0,
            "presence_counts": {},
            "positive_rate": None,
            "date_min": None,
            "date_max": None,
        }
    y = pd.to_numeric(df["Presence"], errors="coerce").fillna(0).astype(int)
    return {
        "rows": int(len(df)),
        "presence_counts": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "positive_rate": float((y == 1).mean()),
        "date_min": str(df["SIGHTINGDATE"].min()) if "SIGHTINGDATE" in df.columns else None,
        "date_max": str(df["SIGHTINGDATE"].max()) if "SIGHTINGDATE" in df.columns else None,
    }


def generate_all(input_csv: str, output_dir: str, seed: int = 42) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    variants: List[Variant] = [
        Variant("acoustic_all_years", include_visual=False),
        Variant("acoustic_2019plus", include_visual=False, min_year=2019),
        Variant("mixed_all_years", include_visual=True),
        Variant("mixed_2019plus", include_visual=True, min_year=2019),
    ]

    manifest: Dict[str, object] = {
        "input_csv": input_csv,
        "output_dir": output_dir,
        "seed": seed,
        "variants": {},
        "notes": [
            "All files are cleaned, deduplicated, and training-ready.",
            "Imputed files fill env NaN values with medians and keep observation flags.",
            "Balanced files are for prototyping only; evaluate on natural class distributions.",
        ],
    }

    for v in variants:
        vdir = os.path.join(output_dir, v.name)
        os.makedirs(vdir, exist_ok=True)

        prepared = _prepare_variant(
            input_csv=input_csv,
            out_dir=vdir,
            include_visual=v.include_visual,
            min_year=v.min_year,
            max_year=v.max_year,
        )

        imp_df = _read_csv(prepared["imputed"])

        bal_1to1 = _balance_undersample(imp_df, ratio_neg_to_pos=1.0, seed=seed)
        bal_2to1 = _balance_undersample(imp_df, ratio_neg_to_pos=2.0, seed=seed)

        bal_1to1_path = os.path.join(vdir, "training_ready_balanced_neg1_pos1.csv")
        bal_2to1_path = os.path.join(vdir, "training_ready_balanced_neg2_pos1.csv")
        bal_1to1.to_csv(bal_1to1_path, index=False)
        bal_2to1.to_csv(bal_2to1_path, index=False)

        manifest["variants"][v.name] = {
            "config": {
                "include_visual": v.include_visual,
                "min_year": v.min_year,
                "max_year": v.max_year,
            },
            "files": {
                "raw": prepared["raw"],
                "imputed": prepared["imputed"],
                "summary": prepared["summary"],
                "balanced_neg1_pos1": bal_1to1_path,
                "balanced_neg2_pos1": bal_2to1_path,
            },
            "stats": {
                "imputed": _stats(imp_df),
                "balanced_neg1_pos1": _stats(bal_1to1),
                "balanced_neg2_pos1": _stats(bal_2to1),
            },
        }

    manifest_path = os.path.join(output_dir, "dataset_variants_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    notes_path = os.path.join(output_dir, "dataset_variants_notes.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("# Dataset variants generated\n\n")
        f.write("This folder contains multiple cleaned variants for model-design discussion.\n\n")
        for name, meta in manifest["variants"].items():
            st = meta["stats"]["imputed"]
            f.write(f"## {name}\n")
            f.write(f"- rows: {st['rows']}\n")
            f.write(f"- positive rate: {st['positive_rate']:.4f}\n" if st["positive_rate"] is not None else "- positive rate: n/a\n")
            f.write(f"- date range: {st['date_min']} -> {st['date_max']}\n")
            f.write(f"- imputed: {meta['files']['imputed']}\n")
            f.write(f"- balanced (1:1): {meta['files']['balanced_neg1_pos1']}\n")
            f.write(f"- balanced (2:1): {meta['files']['balanced_neg2_pos1']}\n\n")

        f.write("## Recommended baseline\n")
        f.write("- acoustic_all_years/training_ready_acoustic_imputed.csv\n")
        f.write("\n## Important caution\n")
        f.write("- Balanced files are not final evaluation sets. Keep natural imbalance in holdout tests.\n")

    return {
        "manifest": manifest_path,
        "notes": notes_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all cleaned dataset variants")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = generate_all(args.input, args.output_dir, seed=args.seed)
    print("Generated:")
    for k, v in out.items():
        print(f"- {k}: {v}")
