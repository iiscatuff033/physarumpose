import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    valid = df[df['strong_anchor_slime_status'] == 'ok'].copy()
    if len(valid) == 0:
        print('No valid rows')
        return

    summary = pd.DataFrame([{
        'count': len(valid),
        'yolo_mean_px': valid['yolo_target_error_px'].mean(),
        'strong_anchor_slime_mean_px': valid['strong_anchor_slime_error_px'].mean(),
        'yolo_median_px': valid['yolo_target_error_px'].median(),
        'strong_anchor_slime_median_px': valid['strong_anchor_slime_error_px'].median(),
        'win_vs_yolo': valid['strong_anchor_slime_better_than_yolo'].mean(),
        'mean_anchor_err': valid['mean_anchor_err'].mean() if 'mean_anchor_err' in valid.columns else None,
        'median_anchor_err': valid['mean_anchor_err'].median() if 'mean_anchor_err' in valid.columns else None,
    }])
    summary.to_csv(out_dir / 'strong_anchor_slime_summary.csv', index=False)

    by_target = valid.groupby('target_name').agg(
        count=('strong_anchor_slime_error_px', 'count'),
        yolo_mean_px=('yolo_target_error_px', 'mean'),
        strong_anchor_slime_mean_px=('strong_anchor_slime_error_px', 'mean'),
        yolo_median_px=('yolo_target_error_px', 'median'),
        strong_anchor_slime_median_px=('strong_anchor_slime_error_px', 'median'),
        win_vs_yolo=('strong_anchor_slime_better_than_yolo', 'mean'),
        mean_anchor_err=('mean_anchor_err', 'mean'),
    ).reset_index()
    by_target.to_csv(out_dir / 'strong_anchor_slime_summary_by_target.csv', index=False)

    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-150', '150+']
    valid['yolo_error_bin'] = pd.cut(valid['yolo_target_error_px'], bins=bins, labels=labels, include_lowest=True, right=False)
    error_bins = valid.groupby('yolo_error_bin', observed=False).agg(
        count=('strong_anchor_slime_error_px', 'count'),
        yolo_mean_px=('yolo_target_error_px', 'mean'),
        strong_anchor_slime_mean_px=('strong_anchor_slime_error_px', 'mean'),
        yolo_median_px=('yolo_target_error_px', 'median'),
        strong_anchor_slime_median_px=('strong_anchor_slime_error_px', 'median'),
        win_vs_yolo=('strong_anchor_slime_better_than_yolo', 'mean'),
    ).reset_index()
    error_bins.to_csv(out_dir / 'strong_anchor_slime_error_bins.csv', index=False)

    status = df['strong_anchor_slime_status'].value_counts().reset_index()
    status.columns = ['status', 'count']
    status.to_csv(out_dir / 'strong_anchor_slime_status_summary.csv', index=False)

    print('\nSummary:')
    print(summary)
    print('\nBy target:')
    print(by_target)
    print('\nBy YOLO error bin:')
    print(error_bins)


if __name__ == '__main__':
    main()
