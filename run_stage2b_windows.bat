@echo off
echo PhysarumPose Stage 2B: YOLO Pose + Skeleton Prior Correction

python 12_run_yolo_pose_on_synth.py --csv_path data/processed/synth_occlusion_mpii/synth_occlusion_metadata.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage2b_yolo_pose --yolo_model yolov8n-pose.pt --max_samples 2000

python 13_correct_yolo_with_skeleton_prior.py --yolo_csv outputs/stage2b_yolo_pose/yolo_pose_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --prior_model_dir outputs/random_mask_prior_mpii_finetuned --out_dir outputs/stage2b_prior_correction

echo Done.
pause
