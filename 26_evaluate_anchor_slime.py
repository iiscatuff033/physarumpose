import argparse
from pathlib import Path

import pandas as pd


def add_bins(df):
    if "yolo_target_error_px" not in df.columns:
        return df
    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"]
    df = df.copy()
    df["yolo_error_bin"] = pd.cut(df["yolo_target_error_px"], bins=bins, labels=labels, include_lowest=True, right=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results_csv)
    valid = df[df["anchor_slime_status"] == "ok"].copy()

    if len(valid) == 0:
        print("No valid rows.")
        return

    summary = {
        "count": len(valid),
        "anchor_slime_mean_px": valid["anchor_slime_error_px"].mean(),
        "anchor_slime_median_px": valid["anchor_slime_error_px"].median(),
    }
    if "yolo_target_error_px" in valid.columns:
        summary["yolo_mean_px"] = valid["yolo_target_error_px"].mean()
        summary["yolo_median_px"] = valid["yolo_target_error_px"].median()
        summary["anchor_slime_win_vs_yolo"] = valid["anchor_slime_better_than_yolo"].mean()
    if "slime_error_px" in valid.columns:
        summary["yolo_anchor_slime_mean_px"] = valid["slime_error_px"].mean()
    if "prior_error_px" in valid.columns:
        summary["prior_mean_px"] = valid["prior_error_px"].mean()
    if "geometry_error_px" in valid.columns:
        summary["geometry_mean_px"] = valid["geometry_error_px"].mean()

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "anchor_slime_summary.csv", index=False)

    agg_dict = {
        "count": ("anchor_slime_error_px", "count"),
        "anchor_slime_mean_px": ("anchor_slime_error_px", "mean"),
        "anchor_slime_median_px": ("anchor_slime_error_px", "median"),
        "anchor_min_conf_mean": ("anchor_min_conf_used", "mean"),
    }
    if "yolo_target_error_px" in valid.columns:
        agg_dict["yolo_mean_px"] = ("yolo_target_error_px", "mean")
        agg_dict["yolo_median_px"] = ("yolo_target_error_px", "median")
        agg_dict["anchor_slime_win_vs_yolo"] = ("anchor_slime_better_than_yolo", "mean")
    by_target = valid.groupby("target_name").agg(**agg_dict).reset_index()
    by_target.to_csv(out_dir / "anchor_slime_summary_by_target.csv", index=False)

    status = df["anchor_slime_status"].value_counts().reset_index()
    status.columns = ["anchor_slime_status", "count"]
    status.to_csv(out_dir / "anchor_slime_status_summary.csv", index=False)

    if "yolo_target_error_px" in valid.columns:
        binned = add_bins(valid)
        bins = binned.groupby("yolo_error_bin", observed=False).agg(
            count=("anchor_slime_error_px", "count"),
            yolo_mean_px=("yolo_target_error_px", "mean"),
            anchor_slime_mean_px=("anchor_slime_error_px", "mean"),
            yolo_median_px=("yolo_target_error_px", "median"),
            anchor_slime_median_px=("anchor_slime_error_px", "median"),
            anchor_slime_win_vs_yolo=("anchor_slime_better_than_yolo", "mean"),
        ).reset_index()
        bins.to_csv(out_dir / "anchor_slime_error_bins.csv", index=False)
        print("\nError bins:")
        print(bins)

    print("\nSummary:")
    print(summary_df)
    print("\nBy target:")
    print(by_target)
    print(f"\nSaved summaries to: {out_dir}")


if __name__ == "__main__":
    main()
