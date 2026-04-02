# Data Collection Quickstart

Use this guide to run the whale dataset build with environment caching.

## What this script does
`build_real_whale_dataset.py` combines:
- acoustic detections (local files under `data/raw/acoustic`)
- optional visual sightings (`otherData/23305_RWSAS.csv`)
- optional environmental variables fetched from ERDDAP (SST, chlorophyll, salinity, etc.)

It writes:
- a dataset CSV
- a run report JSON
- an environment lookup cache CSV (reused in future runs)

---

## 1) One-time setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Cache location (important)

By default, cache files are repo-relative in `data/processed/`:
- paper mode: `data/processed/env_lookup_cache_paper.csv`
- coastwatch mode: `data/processed/env_lookup_cache_coastwatch.csv`

If you already have a cache file, copy it there (or pass `--env-cache <path>`).

---

## 3) Recommended run command (paper mode)

```bash
python build_real_whale_dataset.py \
  --mode paper \
  --random-run-dir \
  --sleep 0.0 \
  --checkpoint-every 1 \
  --quiet
```

This creates output in:
- `data/runs/run_<timestamp>_<hex>/whale_dataset_paper.csv`
- `data/runs/run_<timestamp>_<hex>/report_paper.json`

---

## 4) Fast test command (no API calls)

Use this to verify the pipeline works without network fetching:

```bash
python build_real_whale_dataset.py --no-env --max-rows 50 --random-run-dir --quiet
```

---

## 5) Useful flags

- `--mode paper|coastwatch` → choose data source strategy
- `--random-run-dir` → keep each run isolated and reproducible
- `--env-cache <path>` → use a specific cache file
- `--checkpoint-every N` → save cache progress every N points
- `--no-env` → disable environmental API fetches
- `--no-visual` → ignore visual sightings file
- `--max-rows N` → small test run

---

## 6) Common issues and fixes

- **Many missing env values:** normal for some dates/locations; script keeps rows and marks missing flags.
- **Slow run:** expected with API calls; cache reuse makes later runs faster.
- **Timeouts/HTTP errors:** retry later, or run with `--no-env` for local pipeline validation.
- **Wrong Python env:** re-activate venv with `source venv/bin/activate`.

---

## 7) Minimal smoke-test flow

```bash
source venv/bin/activate
python build_real_whale_dataset.py --no-env --max-rows 50 --random-run-dir --quiet
ls -1 data/runs | tail -n 1
```

This quickly proves the script runs and writes outputs.
