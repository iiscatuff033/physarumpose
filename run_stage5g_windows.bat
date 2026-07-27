@echo off
echo Stage 5G: Target-person crop and wrong-instance filtering

python 60_create_target_person_crop_metadata.py --gated_csv outputs/stage5fc_anchor_quality_gate/gated_predictions.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5g_target_person_crop_filter --pad 120

python 61_visualize_target_person_crops.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5g_target_person_crop_filter/visualizations --max_images 200

python 61_visualize_target_person_crops.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5g_target_person_crop_filter/presentation_images --max_images 80 --presentation_clean --only_good

python 61_visualize_target_person_crops.py --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5g_target_person_crop_filter/suspicious_cases --max_images 100 --only_suspicious

python 62_make_stage5g_report_text.py --crop_summary_csv outputs/stage5g_target_person_crop_filter/crop_quality_summary.csv --crop_csv outputs/stage5g_target_person_crop_filter/crop_metadata.csv --out_txt outputs/stage5g_target_person_crop_filter/stage5g_report_text.txt

echo Done.
pause
