@echo off
cd /d "%~dp0"
call venv_gpu\Scripts\activate
for %%S in (1 2) do (
  set SEED=%%S
  for %%M in (EfficientNet-B0 MobileNetV2 MobileNetV3-Small ResNet50 EfficientNetV2-S) do (
    if not exist out\cnn_%%M_clean_s%%S.json (
      echo ==== %%M clean seed %%S ====
      python train_torch.py %%M clean
    )
  )
)
set SEED=
echo LISTO
