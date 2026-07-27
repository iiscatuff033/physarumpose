@echo off
echo Stage 5E: Multi-Hypothesis Slime Head

python 48_evaluate_multihypothesis_slime.py --candidate_csv outputs/stage5d_coarse_region_slime_scorer/all_candidate_predictions.csv --out_dir outputs/stage5e_multihypothesis_slime --top_k 5 --nms_dist 12 --temperature 0.10

python 49_visualize_multihypothesis_slime.py --topk_csv outputs/stage5e_multihypothesis_slime/topk_predictions_long.csv --candidate_csv outputs/stage5d_coarse_region_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5e_multihypothesis_slime/visualizations --max_images 150

python 49_visualize_multihypothesis_slime.py --topk_csv outputs/stage5e_multihypothesis_slime/topk_predictions_long.csv --candidate_csv outputs/stage5d_coarse_region_slime_scorer/all_candidate_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5e_multihypothesis_slime/presentation_images --max_images 40 --presentation_clean

python 50_make_multihypothesis_report_text.py --summary_csv outputs/stage5e_multihypothesis_slime/multihypothesis_summary.csv --by_target_csv outputs/stage5e_multihypothesis_slime/multihypothesis_summary_by_target.csv --out_txt outputs/stage5e_multihypothesis_slime/stage5e_report_text.txt

echo Done.
pause
