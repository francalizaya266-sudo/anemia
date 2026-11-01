@echo off
rem Reejecucion completa de las CNN sobre el conjunto ESTRICTO (452 imagenes; meta_groups_strict.csv).
rem Crea la subcarpeta strict\ con copias de los scripts y usa el mismo dataset y entorno virtual.
cd /d "%~dp0"
if not exist strict mkdir strict
if not exist strict\out mkdir strict\out
copy /Y common.py strict\ >nul
copy /Y train_torch.py strict\ >nul
copy /Y train_torch_grouped.py strict\ >nul
copy /Y gradcam_torch.py strict\ >nul
copy /Y gradcam_review.py strict\ >nul
copy /Y meta_groups_strict.csv strict\meta_groups.csv >nul
call venv_gpu\Scripts\activate
set DATA_DIR=%~dp0cp-anemic
cd strict
set SEED=42
for %%M in (MobileNetV3-Small MobileNetV2 MobileNetV3-Large EfficientNet-B0 ShuffleNetV2 EfficientNetV2-S ResNet50) do (
  if not exist out\cnn_%%M_clean.json (
    echo ==== STRICT %%M clean ====
    python train_torch.py %%M clean
  )
)
for %%M in (MobileNetV3-Small MobileNetV2 EfficientNet-B0) do (
  if not exist out\cnn_%%M_naive.json (
    echo ==== STRICT %%M naive ====
    python train_torch.py %%M naive
  )
)
for %%M in (EfficientNet-B0 MobileNetV2 MobileNetV3-Small MobileNetV3-Large ShuffleNetV2 ResNet50 EfficientNetV2-S) do (
  if not exist out\cnn_%%M_grouped.json (
    echo ==== STRICT %%M grouped ====
    python train_torch_grouped.py %%M
  )
)
for %%S in (1 2) do (
  set SEED=%%S
  for %%M in (EfficientNet-B0 MobileNetV2 MobileNetV3-Small ResNet50 EfficientNetV2-S) do (
    if not exist out\cnn_%%M_clean_s%%S.json (
      echo ==== STRICT %%M clean seed %%S ====
      python train_torch.py %%M clean
    )
  )
)
set SEED=42
if not exist out\gradcam_review.json (
  echo ==== STRICT Grad-CAM 5 folds ====
  python gradcam_review.py
)
echo LISTO_STRICT
