@echo off
echo Stage 4B: Strong AnchorNet + Slime

python 28_prepare_anchor_heatmap_dataset.py --mat_path data/raw/mpii_human_pose_v1_u12_1.mat --image_dir data/raw/images --out_csv data/processed/anchor_heatmap_mpii/anchor_heatmap_dataset.csv --min_visible_anchors 5

python 29_train_strong_anchor_net.py --csv_path data/processed/anchor_heatmap_mpii/anchor_heatmap_dataset.csv --image_dir data/raw/images --out_dir outputs/strong_anchor_net_mpii --epochs 80 --batch_size 24 --input_h 384 --input_w 288

python 30_predict_strong_anchor_net_on_synth.py --metadata_csv outputs/stage2b_prior_correction/yolo_prior_correction_results.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/strong_anchor_net_mpii --out_dir outputs/stage4b_strong_anchor_predictions --crop_mode gt_bbox

python 31_slime_path_with_strong_anchor_net.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage4b_anchor_slime --max_samples 2000

python 32_evaluate_strong_anchor_slime.py --results_csv outputs/stage4b_anchor_slime/strong_anchor_slime_results.csv --out_dir outputs/stage4b_anchor_slime

python 33_visualize_strong_anchor_slime.py --results_csv outputs/stage4b_anchor_slime/strong_anchor_slime_results.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage4b_anchor_slime/visualizations --only_valid

echo Done.
pause
