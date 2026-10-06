# PhysarumPose Stage 1: Body-Structure Prior on MPII

<img width="1920" height="1080" alt="video_03_124 BO Swetha G_5160079_A_frame_0461_GOOD_score6 15_t53 60_idx1340_bfaaf4b0-7929-451a-bb2e-d8a9a5f11777_blueGT_redDARK_greenUDP" src="https://github.com/user-attachments/assets/56665fa7-e99e-4efc-b963-401f51b76587" />


This is the first training stage. We do not train an image model yet.

We train a coordinate-only model that learns simple body geometry:

- shoulder + wrist -> elbow
- hip + ankle -> knee

## Model used

SkeletonPriorMLP.

Input:
- endpoint 1: x, y
- endpoint 2: x, y
- task one-hot vector

Output:
- missing middle joint: x, y

## Dataset

Use MPII Human Pose annotation file:

data/raw/mpii_human_pose_v1_u12_1.mat

Images are not required for this first stage.

## Install

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Prepare clean visible skeleton samples

python 01_prepare_mpii_clean_skeletons.py --mat_path data/raw/mpii_human_pose_v1_u12_1.mat --out_csv data/processed/mpii_clean_limb_tasks.csv

## Train

python 02_train_skeleton_prior.py --csv_path data/processed/mpii_clean_limb_tasks.csv --out_dir outputs/skeleton_prior_mpii --epochs 100 --batch_size 256

## Evaluate

python 03_evaluate_skeleton_prior.py --csv_path outputs/skeleton_prior_mpii/val_split.csv --model_dir outputs/skeleton_prior_mpii --out_dir outputs/eval_plots
"# physarumpose" 
