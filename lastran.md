python 94_evaluate_stage6e_single_model.py --crop_csv outputs/stage6e_large_context_crops/crop_metadata.csv --annotation_csv outputs/stage4b_strong_anchor_predictions/strong_anchor_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --model_dir outputs/stage6e_single_model_semantic_offset --out_dir outputs/stage6e_single_model_semantic_offset/eval --batch_size 8 --input_size 384



python 95_visualize_stage6e_single_model.py --pred_csv outputs/stage6e_single_model_semantic_offset/eval/stage6e_predictions.csv --topk_csv outputs/stage6e_single_model_semantic_offset/eval/stage6e_topk_predictions.csv --image_dir data/processed/synth_occlusion_mpii/images --out_dir outputs/stage6e_single_model_semantic_offset/eval/visualizations --max_images 150


python 96_make_stage6e_report.py --summary_csv outputs/stage6e_single_model_semantic_offset/eval/stage6e_summary.csv --by_target_csv outputs/stage6e_single_model_semantic_offset/eval/stage6e_summary_by_target.csv --out_txt outputs/stage6e_single_model_semantic_offset/eval/stage6e_report_text.txt