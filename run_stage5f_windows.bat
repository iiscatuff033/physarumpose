@echo off
echo Stage 5F: End-to-end differentiable soft-slime model

python 51_train_e2e_soft_slime.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5f_e2e_soft_slime --epochs 80 --batch_size 16 --input_size 256 --max_samples 2000

python 52_evaluate_e2e_soft_slime.py --csv_path outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage5f_e2e_soft_slime --out_dir outputs/stage5f_e2e_soft_slime/eval --max_samples 2000 --batch_size 16

python 53_visualize_e2e_soft_slime.py --pred_csv outputs/stage5f_e2e_soft_slime/eval/e2e_predictions.csv --topk_csv outputs/stage5f_e2e_soft_slime/eval/e2e_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5f_e2e_soft_slime/eval/visualizations --max_images 150

python 53_visualize_e2e_soft_slime.py --pred_csv outputs/stage5f_e2e_soft_slime/eval/e2e_predictions.csv --topk_csv outputs/stage5f_e2e_soft_slime/eval/e2e_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage5f_e2e_soft_slime/eval/presentation_images --max_images 50 --presentation_clean

echo Done.
pause
