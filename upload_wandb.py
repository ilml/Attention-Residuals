"""
Upload merged wandb runs from offline data.
Reads all offline runs, groups by experiment name, merges full history,
and uploads one clean run per experiment with all metrics.

Usage:
  WANDB_API_KEY=... python upload_wandb.py --entity tom-vita --project attenres
"""

import os
import json
import glob
import argparse
from collections import defaultdict

import wandb
import pandas as pd


def load_offline_run(run_dir):
    """Load history and config from an offline wandb run directory."""
    files_dir = os.path.join(run_dir, "files")

    # Load config
    config = {}
    config_path = os.path.join(files_dir, "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        # wandb config has nested {"value": x} format
        for k, v in raw.items():
            if isinstance(v, dict) and "value" in v:
                config[k] = v["value"]
            else:
                config[k] = v

    # Load history from jsonl
    history = []
    hist_path = os.path.join(run_dir, "run-history.jsonl")
    if not os.path.exists(hist_path):
        # Try alternative path
        for candidate in glob.glob(os.path.join(files_dir, "wandb-history.jsonl")):
            hist_path = candidate
            break

    if os.path.exists(hist_path):
        with open(hist_path) as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    history.append(row)
                except json.JSONDecodeError:
                    continue

    # Get run name from wandb-metadata.json
    name = None
    meta_path = os.path.join(files_dir, "wandb-metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            name = meta.get("name") or meta.get("codePath")

    # Also try from run config
    if not name and "model_config" in config and "variant" in config:
        name = f"{config['model_config']}_{config['variant']}"

    return name, config, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="tom-vita")
    parser.add_argument("--project", type=str, default="attenres")
    parser.add_argument("--wandb_dir", type=str, default="wandb")
    args = parser.parse_args()

    # Find all offline runs
    offline_dirs = sorted(glob.glob(os.path.join(args.wandb_dir, "offline-run-*")))
    online_dirs = sorted(glob.glob(os.path.join(args.wandb_dir, "run-*")))
    all_dirs = offline_dirs + online_dirs
    print(f"Found {len(all_dirs)} run directories")

    # Load and group by experiment name
    experiments = defaultdict(lambda: {"configs": [], "histories": []})

    for run_dir in all_dirs:
        name, config, history = load_offline_run(run_dir)
        if not name or not history:
            continue
        experiments[name]["configs"].append(config)
        experiments[name]["histories"].extend(history)

    print(f"Found {len(experiments)} unique experiments with data\n")

    # Upload each experiment as one merged run
    for name in sorted(experiments.keys()):
        data = experiments[name]
        all_rows = data["histories"]
        config = data["configs"][-1] if data["configs"] else {}

        # Build dataframe and deduplicate by step
        df = pd.DataFrame(all_rows)
        if "_step" in df.columns:
            df = df.sort_values("_step").drop_duplicates(subset=["_step"], keep="last")
        elif "train/step" in df.columns:
            df = df.sort_values("train/step").drop_duplicates(
                subset=["train/step"], keep="last")

        # Extract group info
        parts = name.split("_", 1)
        model_size = parts[0] if len(parts) > 0 else name
        variant = parts[1] if len(parts) > 1 else "unknown"

        # Count metrics
        metric_cols = [c for c in df.columns if not c.startswith("_")]

        print(f"{name}: {len(df)} steps, {len(metric_cols)} metrics: {metric_cols[:8]}...")

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=name,
            group=model_size,
            tags=[variant, model_size, "merged"],
            config=config,
            reinit=True,
        )
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")
        wandb.define_metric("perf/*", step_metric="train/step")

        for _, row in df.iterrows():
            log_dict = {}
            for col in row.index:
                if col.startswith("_") or pd.isna(row[col]):
                    continue
                log_dict[col] = row[col]
            step = int(log_dict.get("train/step", log_dict.get("_step", 0)))
            wandb.log(log_dict, step=step)

        wandb.finish()
        print(f"  -> {run.url}\n")

    print("Done!")


if __name__ == "__main__":
    main()
