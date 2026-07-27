@echo off
echo Stage 6A: Unified Differentiable PhysarumPose

python 68_train_unified_physarum_pose.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6a_unified_physarum_pose --epochs 100 --batch_size 16 --input_size 256

python 69_evaluate_unified_physarum_pose.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage6a_unified_physarum_pose --out_dir outputs/stage6a_unified_physarum_pose/eval --batch_size 16

python 70_visualize_unified_physarum_pose.py --pred_csv outputs/stage6a_unified_physarum_pose/eval/unified_predictions.csv --topk_csv outputs/stage6a_unified_physarum_pose/eval/unified_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6a_unified_physarum_pose/eval/visualizations --max_images 150

python 70_visualize_unified_physarum_pose.py --pred_csv outputs/stage6a_unified_physarum_pose/eval/unified_predictions.csv --topk_csv outputs/stage6a_unified_physarum_pose/eval/unified_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6a_unified_physarum_pose/eval/presentation_images --max_images 80 --only_good

python 71_make_stage6a_report_text.py --summary_csv outputs/stage6a_unified_physarum_pose/eval/unified_summary.csv --by_target_csv outputs/stage6a_unified_physarum_pose/eval/unified_summary_by_target.csv --out_txt outputs/stage6a_unified_physarum_pose/eval/stage6a_report_text.txt

echo Done.
pause
