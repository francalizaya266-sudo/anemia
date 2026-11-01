@echo off
rem ResNet50 (the model of the original CP-AnemiC paper) under the naive protocol: 708 rows, image-level 5-fold CV,
rem seed 42, the same recipe as every other CNN. Its grouped counterpart is out_review\cnn_ResNet50_grouped.json.
rem Output: out\cnn_ResNet50_naive.json (a few minutes on an RTX 4060 Laptop). Copy it back to the project.
cd /d "%~dp0"
call venv_gpu\Scripts\activate
set SEED=42
if not exist out mkdir out
if not exist out\cnn_ResNet50_naive.json (
  echo ==== ResNet50 naive ====
  python train_torch.py ResNet50 naive
)
echo LISTO_RESNET50_NAIVE
