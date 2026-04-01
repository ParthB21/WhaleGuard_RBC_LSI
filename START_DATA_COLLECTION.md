# Starting Data Collection (using env cache)

This document describes a quick flow to start building the real whale dataset using the existing environment lookup cache.

Prerequisites
- Python 3.8+ installed on your system.
- A working internet connection for ERDDAP queries (unless you use `--no-env`).

1) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy pandas requests
# Optionally save installed packages
pip freeze > requirements.txt
```

2) Prepare the repository and cache
- Ensure `data/processed` exists. The script will default to using an env cache at:

- `data/processed/env_lookup_cache_coastwatch.csv` (coastwatch mode)
- `data/processed/env_lookup_cache_paper.csv` (paper mode)

If you already have a cache CSV, place it in `data/processed/` or pass it explicitly with `--env-cache`.

```bash
mkdir -p data/processed
# example: copy your cache file into place
# cp /path/to/your/env_lookup_cache_paper.csv data/processed/
```

3) Run the data build (example)

The command you used before works well; here it is again:

```bash
python build_real_whale_dataset.py \
  --mode paper \
  --random-run-dir \
  --sleep 0.0 \
  --checkpoint-every 1 \
  --quiet
```

Notes and useful flags
- `--mode {coastwatch|paper}`: choose CoastWatch (default) or the UDel "paper" ladder.
- `--random-run-dir`: write outputs to `data/runs/run_<timestamp>_<hex>/` (recommended for experiments).
- `--env-cache PATH`: explicitly point to an env cache CSV (overrides the default cache path).
- `--no-env`: skip all environmental API requests (fast; useful for offline testing).
- `--no-visual`: exclude the visual sightings CSV from the combined dataset.
- `--max-rows N`: limit rows for quick tests.

How the cache is used
- The script derives the default cache path relative to the script root: `data/processed/env_lookup_cache_<mode>.csv`.
- On runs, new environment keys are appended to the cache and periodically checkpointed using `--checkpoint-every`.
- To resume a partial run, pass the same `--env-cache` path so previously-fetched points are reused.

Troubleshooting
- If you see many HTTP errors, check network connectivity or run with `--no-env` to skip ERDDAP fetches.
- If the script creates many `data/runs` outputs, remove old runs with `rm -rf data/runs/run_*` if you want to clean up.

That's it — let me know if you want this added to a full `README.md` or if you'd like a `requirements.txt` generated and committed.
