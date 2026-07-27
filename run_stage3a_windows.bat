@echo off
echo Stage 3A: Slime-style candidate path recovery

python 19_slime_path_candidate_recovery.py --csv_path outputs/stage2b_prior_correction/yolo_prior_correction_results.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage3a_slime_path --max_samples 2000

python 20_evaluate_slime_path_results.py --results_csv outputs/stage3a_slime_path/slime_path_results.csv --out_dir outputs/stage3a_slime_path

python 21_visualize_slime_path_results.py --results_csv outputs/stage3a_slime_path/slime_path_results.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage3a_slime_path/visualizations --only_valid

python 21_visualize_slime_path_results.py --results_csv outputs/stage3a_slime_path/slime_path_results.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage3a_slime_path/visualizations_improved --only_improved

echo Done.
pause
