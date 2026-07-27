@echo off
echo Stage 5F-C: Anchor-quality gated inference

python 57_anchor_quality_gated_inference.py --pred_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_predictions.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --out_dir outputs/stage5fc_anchor_quality_gate

python 58_visualize_anchor_quality_gate.py --gated_csv outputs/stage5fc_anchor_quality_gate/gated_predictions.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5fc_anchor_quality_gate/visualizations --max_images 150

python 58_visualize_anchor_quality_gate.py --gated_csv outputs/stage5fc_anchor_quality_gate/gated_predictions.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5fc_anchor_quality_gate/presentation_images --max_images 50 --presentation_clean --crop_around_path

python 59_make_gate_report_text.py --summary_csv outputs/stage5fc_anchor_quality_gate/gated_summary.csv --by_target_csv outputs/stage5fc_anchor_quality_gate/gated_summary_by_target.csv --out_txt outputs/stage5fc_anchor_quality_gate/stage5fc_report_text.txt

echo Done.
pause
