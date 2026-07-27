@echo off
echo Stage 5B: Learned slime scorer without occlusion-box features

python 37_build_no_occ_anchor_slime_candidate_dataset.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5b_no_occ_candidates --max_samples 2000

python 38_train_no_occ_slime_scorer.py --candidate_csv outputs/stage5b_no_occ_candidates/no_occ_candidate_dataset.csv --feature_json outputs/stage5b_no_occ_candidates/no_occ_feature_names.json --out_dir outputs/stage5b_no_occ_slime_scorer --epochs 80 --batch_size 4096

python 39_visualize_no_occ_slime_results.py --results_csv outputs/stage5b_no_occ_slime_scorer/all_best_predictions.csv --candidate_csv outputs/stage5b_no_occ_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5b_no_occ_slime_scorer/visualizations

python 39_visualize_no_occ_slime_results.py --results_csv outputs/stage5b_no_occ_slime_scorer/all_best_predictions.csv --candidate_csv outputs/stage5b_no_occ_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5b_no_occ_slime_scorer/visualizations_improved --only_improved

echo Done.
pause
