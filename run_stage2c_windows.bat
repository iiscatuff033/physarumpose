@echo off
echo Stage 2C: Trust-aware correction

python 14_grid_search_gated_correction.py --csv_path outputs/stage2b_prior_correction/yolo_prior_correction_results.csv --out_dir outputs/stage2c_gated_correction

python 15_train_correction_head.py --csv_path outputs/stage2b_prior_correction/yolo_prior_correction_results.csv --out_dir outputs/stage2c_correction_head --epochs 120 --batch_size 256

python 16_visualize_correction_head_results.py --results_csv outputs/stage2c_correction_head/val_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage2c_correction_head/visualizations

echo Done.
pause
