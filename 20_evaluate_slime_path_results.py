import argparse
from pathlib import Path
import pandas as pd


def summarize(df):
    out = {
        'count': len(df),
        'yolo_mean_px': df['yolo_target_error_px'].mean(),
        'slime_mean_px': df['slime_error_px'].mean(),
        'yolo_median_px': df['yolo_target_error_px'].median(),
        'slime_median_px': df['slime_error_px'].median(),
        'slime_win_vs_yolo': df['slime_better_than_yolo'].mean(),
    }
    if 'prior_error_px' in df.columns:
        out['prior_mean_px'] = df['prior_error_px'].mean()
        out['slime_win_vs_prior'] = df['slime_better_than_prior'].mean()
    if 'geometry_error_px' in df.columns:
        out['geometry_mean_px'] = df['geometry_error_px'].mean()
        out['slime_win_vs_geometry'] = df['slime_better_than_geometry'].mean()
    return out


def add_error_bins(df):
    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-150', '150+']
    df = df.copy()
    df['yolo_error_bin'] = pd.cut(df['yolo_target_error_px'], bins=bins, labels=labels, include_lowest=True, right=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results_csv)
    valid = df[df['slime_status'] == 'ok'].copy()
    if len(valid) == 0:
        print('No valid slime predictions.')
        return
    summary = pd.DataFrame([{'split': 'valid', **summarize(valid)}])
    summary.to_csv(out_dir / 'slime_path_summary.csv', index=False)
    by_target = valid.groupby('target_name').agg(
        count=('slime_error_px', 'count'),
        yolo_mean_px=('yolo_target_error_px', 'mean'),
        slime_mean_px=('slime_error_px', 'mean'),
        prior_mean_px=('prior_error_px', 'mean'),
        geometry_mean_px=('geometry_error_px', 'mean'),
        yolo_median_px=('yolo_target_error_px', 'median'),
        slime_median_px=('slime_error_px', 'median'),
        slime_win_vs_yolo=('slime_better_than_yolo', 'mean'),
        slime_win_vs_prior=('slime_better_than_prior', 'mean'),
        slime_win_vs_geometry=('slime_better_than_geometry', 'mean'),
    ).reset_index()
    by_target.to_csv(out_dir / 'slime_path_summary_by_target.csv', index=False)
    binned = add_error_bins(valid)
    bins_summary = binned.groupby('yolo_error_bin', observed=False).agg(
        count=('slime_error_px', 'count'),
        yolo_mean_px=('yolo_target_error_px', 'mean'),
        slime_mean_px=('slime_error_px', 'mean'),
        yolo_median_px=('yolo_target_error_px', 'median'),
        slime_median_px=('slime_error_px', 'median'),
        slime_win_vs_yolo=('slime_better_than_yolo', 'mean'),
    ).reset_index()
    bins_summary.to_csv(out_dir / 'slime_path_error_bins.csv', index=False)
    status = df['slime_status'].value_counts().reset_index()
    status.columns = ['slime_status', 'count']
    status.to_csv(out_dir / 'slime_path_status_summary.csv', index=False)
    print('\nOverall summary:')
    print(summary)
    print('\nBy target:')
    print(by_target)
    print('\nBy YOLO error bin:')
    print(bins_summary)


if __name__ == '__main__':
    main()
