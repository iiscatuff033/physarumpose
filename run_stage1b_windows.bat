@echo off
python 04_prepare_mpii_full_context.py --mat_path data/raw/mpii_human_pose_v1_u12_1.mat --out_csv data/processed/mpii_full_context_tasks.csv --min_visible 8
python 05_train_full_context_prior.py --csv_path data/processed/mpii_full_context_tasks.csv --out_dir outputs/full_context_prior_mpii --epochs 120 --batch_size 256
python 06_evaluate_full_context_prior.py --csv_path outputs/full_context_prior_mpii/val_split.csv --model_dir outputs/full_context_prior_mpii --out_dir outputs/eval_plots_full_context
pause
