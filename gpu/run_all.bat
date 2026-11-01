@echo off
cd /d "%~dp0"
call venv_gpu\Scripts\activate
for %%M in (MobileNetV3-Small MobileNetV2 MobileNetV3-Large EfficientNet-B0 ShuffleNetV2 EfficientNetV2-S ResNet50) do (
  echo ==== %%M clean ====
  python train_torch.py %%M clean
)
for %%M in (MobileNetV3-Small MobileNetV2 EfficientNet-B0) do (
  echo ==== %%M naive ====
  python train_torch.py %%M naive
)
python gradcam_torch.py MobileNetV2
python gradcam_torch.py EfficientNet-B0
echo LISTO
