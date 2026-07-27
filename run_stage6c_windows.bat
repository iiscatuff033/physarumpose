@echo off
echo Stage 6C: GT-Anchor Scheduled Unified Differentiable PhysarumPose

python 76_train_gt_anchor_scheduled_physarum_pose.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_gt_anchor_scheduled_physarum_pose --epochs 140 --batch_size 16 --input_size 256 --gt_anchor_warmup_epochs 35 --gt_anchor_ramp_end_epoch 95

python 77_evaluate_gt_anchor_scheduled_physarum_pose.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage6c_gt_anchor_scheduled_physarum_pose --out_dir outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval --batch_size 16

python 78_visualize_gt_anchor_scheduled_physarum_pose.py --pred_csv outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/gt_scheduled_predictions.csv --topk_csv outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/gt_scheduled_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/visualizations --max_images 150

python 78_visualize_gt_anchor_scheduled_physarum_pose.py --pred_csv outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/gt_scheduled_predictions.csv --topk_csv outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/gt_scheduled_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/presentation_images --max_images 80 --only_good

python 79_make_stage6c_report_text.py --summary_csv outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/gt_scheduled_summary.csv --by_target_csv outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/gt_scheduled_summary_by_target.csv --out_txt outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval/stage6c_report_text.txt

echo Done.
pause
