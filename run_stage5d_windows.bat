@echo off
echo Stage 5D: coarse predicted region + learned slime scorer

python 45_build_coarse_region_slime_candidate_dataset.py --csv_path outputs/stage5c_target_region_predictions/target_region_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5d_coarse_region_candidates --max_samples 2000 --cell_px 64

python 46_train_coarse_region_slime_scorer.py --candidate_csv outputs/stage5d_coarse_region_candidates/coarse_region_candidate_dataset.csv --feature_json outputs/stage5d_coarse_region_candidates/coarse_region_feature_names.json --out_dir outputs/stage5d_coarse_region_slime_scorer --epochs 80 --batch_size 4096

python 47_visualize_coarse_region_slime_results.py --results_csv outputs/stage5d_coarse_region_slime_scorer/all_best_predictions.csv --candidate_csv outputs/stage5d_coarse_region_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5d_coarse_region_slime_scorer/visualizations

python 47_visualize_coarse_region_slime_results.py --results_csv outputs/stage5d_coarse_region_slime_scorer/all_best_predictions.csv --candidate_csv outputs/stage5d_coarse_region_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5d_coarse_region_slime_scorer/visualizations_improved_vs_coarse --only_improved_vs_coarse

echo Done.
pause
