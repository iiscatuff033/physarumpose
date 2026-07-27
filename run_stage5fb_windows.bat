@echo off
echo Stage 5F-B: Teacher-anchor differentiable soft-slime

python 54_train_teacher_anchor_soft_slime.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5fb_teacher_anchor_soft_slime --epochs 80 --batch_size 16 --input_size 256 --max_samples 2000

python 55_evaluate_teacher_anchor_soft_slime.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage5fb_teacher_anchor_soft_slime --out_dir outputs/stage5fb_teacher_anchor_soft_slime/eval --max_samples 2000 --batch_size 16

python 56_visualize_teacher_anchor_soft_slime.py --pred_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_predictions.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5fb_teacher_anchor_soft_slime/eval/visualizations --max_images 150

python 56_visualize_teacher_anchor_soft_slime.py --pred_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_predictions.csv --topk_csv outputs/stage5fb_teacher_anchor_soft_slime/eval/teacher_anchor_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5fb_teacher_anchor_soft_slime/eval/presentation_images --max_images 50 --presentation_clean

echo Done.
pause
