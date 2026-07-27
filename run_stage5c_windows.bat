@echo off
echo Stage 5C: Predicted region + learned slime scorer

python 40_train_target_region_net.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5c_target_region_net --epochs 60 --batch_size 24 --input_size 256

python 41_predict_target_region_net.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage5c_target_region_net --out_dir outputs/stage5c_target_region_predictions

python 42_build_pred_region_slime_candidate_dataset.py --csv_path outputs/stage5c_target_region_predictions/target_region_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5c_pred_region_candidates --max_samples 2000

python 43_train_pred_region_slime_scorer.py --candidate_csv outputs/stage5c_pred_region_candidates/pred_region_candidate_dataset.csv --feature_json outputs/stage5c_pred_region_candidates/pred_region_feature_names.json --out_dir outputs/stage5c_pred_region_slime_scorer --epochs 80 --batch_size 4096

python 44_visualize_pred_region_slime_results.py --results_csv outputs/stage5c_pred_region_slime_scorer/all_best_predictions.csv --candidate_csv outputs/stage5c_pred_region_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5c_pred_region_slime_scorer/visualizations

python 44_visualize_pred_region_slime_results.py --results_csv outputs/stage5c_pred_region_slime_scorer/all_best_predictions.csv --candidate_csv outputs/stage5c_pred_region_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5c_pred_region_slime_scorer/visualizations_improved --only_improved

echo Done.
pause
