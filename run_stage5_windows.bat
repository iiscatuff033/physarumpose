@echo off
echo Stage 5: Learned Slime Candidate Scorer with Strong AnchorNet

python 34_build_anchor_slime_candidate_dataset.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5_learned_slime_candidates --max_samples 2000

python 35_train_learned_slime_scorer.py --candidate_csv outputs/stage5_learned_slime_candidates/anchor_slime_candidate_dataset.csv --feature_json outputs/stage5_learned_slime_candidates/candidate_feature_names.json --out_dir outputs/stage5_learned_slime_scorer --epochs 80 --batch_size 4096

python 36_visualize_learned_slime_results.py --results_csv outputs/stage5_learned_slime_scorer/all_best_predictions.csv --candidate_csv outputs/stage5_learned_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5_learned_slime_scorer/visualizations

echo Done.
pause
