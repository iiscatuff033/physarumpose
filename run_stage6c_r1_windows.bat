@echo off
echo Stage 6C-R1: CVPR Anchor-Curriculum Differentiable PhysarumPose

python 80_train_stage6c_r1.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_r1_cvpr_anchor_curriculum --epochs 220 --batch_size 16 --input_size 256 --anchor_pretrain_epochs 50 --gt_anchor_warmup_epochs 80 --gt_anchor_ramp_end_epoch 140 --save_best_after_epoch 140

python 81_evaluate_stage6c_r1.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage6c_r1_cvpr_anchor_curriculum --out_dir outputs/stage6c_r1_cvpr_anchor_curriculum/eval --batch_size 16

python 82_visualize_stage6c_r1.py --pred_csv outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_predictions.csv --topk_csv outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_r1_cvpr_anchor_curriculum/eval/visualizations --max_images 150

python 82_visualize_stage6c_r1.py --pred_csv outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_predictions.csv --topk_csv outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_r1_cvpr_anchor_curriculum/eval/presentation_images --max_images 80 --only_good

python 83_make_stage6c_r1_report.py --summary_csv outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_summary.csv --by_target_csv outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_summary_by_target.csv --out_txt outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_report_text.txt

echo Done.
pause
