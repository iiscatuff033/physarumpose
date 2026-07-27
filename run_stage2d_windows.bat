@echo off
echo Stage 2D: Residual correction + gate

python 17_train_residual_gate_model.py --csv_path outputs/stage2b_prior_correction/yolo_prior_correction_results.csv --out_dir outputs/stage2d_residual_gate --epochs 160 --batch_size 256

python 18_visualize_residual_gate_results.py --results_csv outputs/stage2d_residual_gate/val_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage2d_residual_gate/visualizations

python 18_visualize_residual_gate_results.py --results_csv outputs/stage2d_residual_gate/val_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage2d_residual_gate/visualizations_improved --only_improved

echo Done.
pause
