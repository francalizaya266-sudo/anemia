@echo off
cd /d "%~dp0"
call venv_gpu\Scripts\activate
set SEED=42
for %%M in (EfficientNet-B0 MobileNetV2 MobileNetV3-Small MobileNetV3-Large ShuffleNetV2 ResNet50 EfficientNetV2-S) do (
  if not exist out\cnn_%%M_grouped.json (
    echo ==== %%M grouped ====
    python train_torch_grouped.py %%M
  )
)
if not exist out\gradcam_review.json (
  echo ==== Grad-CAM 5 folds ====
  python gradcam_review.py
)
echo LISTO_REVIEW
