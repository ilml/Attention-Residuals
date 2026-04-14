"""
Merge fragmented wandb runs from preemptions into single continuous runs.

For each experiment (e.g. 194M_baseline), finds all runs with that name,
pulls their history, merges into one new run sorted by step, and deletes
the old fragments.

Usage:
  python merge_wandb_runs.py --entity tom-vita --project attenres [--dry_run]
"""

import argparse
import wandb
import pandas as pd
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="tom-vita")
    parser.add_argument("--project", type=str, default="attenres")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would happen without making changes")
    args = parser.parse_args()

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    print(f"Scanning {project_path}...")

    runs = api.runs(project_path)
    print(f"Found {len(runs)} total runs\n")

    # Group runs by name
    groups = defaultdict(list)
    for run in runs:
        groups[run.name].append(run)

    # Find groups with multiple fragments
    for name, run_list in sorted(groups.items()):
        if len(run_list) <= 1:
            continue

        # Sort fragments by their earliest step
        fragments = []
        for run in run_list:
            # Fetch ALL metrics, not just train/loss
            hist = run.history(samples=10000)
            if hist.empty or "train/step" not in hist.columns:
                fragments.append((run, 0, 0, pd.DataFrame()))
                continue
            min_step = hist["train/step"].min()
            max_step = hist["train/step"].max()
            fragments.append((run, min_step, max_step, hist))

        fragments.sort(key=lambda x: x[1])  # sort by earliest step

        total_points = sum(len(f[3]) for f in fragments)
        step_ranges = [f"[{int(f[1])}-{int(f[2])}]" for f in fragments if len(f[3]) > 0]
        print(f"{name}: {len(fragments)} fragments, {total_points} total points")
        print(f"  Ranges: {', '.join(step_ranges)}")

        if args.dry_run:
            print(f"  [DRY RUN] Would merge into 1 run and delete {len(fragments)} old ones\n")
            continue

        # Merge all history into one dataframe, deduplicate by step
        non_empty = [f[3] for f in fragments if len(f[3]) > 0]
        if not non_empty:
            # All fragments are empty — just delete them all
            print(f"  All fragments empty, deleting {len(fragments)} ghost runs")
            for run, _, _, _ in fragments:
                run.delete()
                print(f"  Deleted: {run.id}")
            print()
            continue
        all_hist = pd.concat(non_empty, ignore_index=True)
        if all_hist.empty:
            print(f"  No data to merge, skipping\n")
            continue

        # Keep last value per step (latest fragment wins)
        all_hist = all_hist.sort_values("train/step").drop_duplicates(
            subset=["train/step"], keep="last").reset_index(drop=True)
        print(f"  Merged: {len(all_hist)} unique steps")

        # Get config from the longest fragment
        longest = max(fragments, key=lambda f: len(f[3]))
        source_run = longest[0]
        config = dict(source_run.config)
        tags = list(source_run.tags) if source_run.tags else []
        group = source_run.group

        # Create a new merged run
        new_run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=name,
            group=group,
            tags=tags + ["merged"],
            config=config,
            reinit=True,
        )
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")
        wandb.define_metric("perf/*", step_metric="train/step")

        # Log all merged data points
        for _, row in all_hist.iterrows():
            log_dict = {}
            for col in row.index:
                if col.startswith("_") or pd.isna(row[col]):
                    continue
                log_dict[col] = row[col]
            if "train/step" in log_dict:
                wandb.log(log_dict, step=int(log_dict["train/step"]))

        wandb.finish()
        print(f"  Created merged run: {new_run.url}")

        # Delete old fragment runs
        for run, _, _, _ in fragments:
            run.delete()
            print(f"  Deleted: {run.id}")

        print()

    print("Done!")


if __name__ == "__main__":
    main()
