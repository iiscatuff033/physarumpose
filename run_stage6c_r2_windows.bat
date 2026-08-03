@echo off
echo Stage 6C-R2: CVPR Anchor-Hardened Differentiable PhysarumPose

python 84_train_stage6c_r2.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_r2_cvpr_anchor_hardened --epochs 260 --batch_size 16 --input_size 256 --anchor_pretrain_epochs 70 --gt_anchor_warmup_epochs 100 --gt_anchor_ramp_end_epoch 160 --save_best_after_epoch 160 --w_anchor_coord 32 --anchor_heatmap_weight 14 --select_anchor_weight 3.5 --select_soft_weight 1.0

python 85_evaluate_stage6c_r2.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage6c_r2_cvpr_anchor_hardened --out_dir outputs/stage6c_r2_cvpr_anchor_hardened/eval --batch_size 16

python 86_visualize_stage6c_r2.py --pred_csv outputs/stage6c_r2_cvpr_anchor_hardened/eval/stage6c_r2_predictions.csv --topk_csv outputs/stage6c_r2_cvpr_anchor_hardened/eval/stage6c_r2_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_r2_cvpr_anchor_hardened/eval/visualizations --max_images 150

python 86_visualize_stage6c_r2.py --pred_csv outputs/stage6c_r2_cvpr_anchor_hardened/eval/stage6c_r2_predictions.csv --topk_csv outputs/stage6c_r2_cvpr_anchor_hardened/eval/stage6c_r2_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_r2_cvpr_anchor_hardened/eval/presentation_images --max_images 80 --only_good

python 87_make_stage6c_r2_report.py --summary_csv outputs/stage6c_r2_cvpr_anchor_hardened/eval/stage6c_r2_summary.csv --by_target_csv outputs/stage6c_r2_cvpr_anchor_hardened/eval/stage6c_r2_summary_by_target.csv --out_txt outputs/stage6c_r2_cvpr_anchor_hardened/eval/stage6c_r2_report_text.txt

echo Done.
pause
