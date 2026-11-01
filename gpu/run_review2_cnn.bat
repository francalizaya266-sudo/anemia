@echo off
rem Segunda revision interna: CNN en GPU. Tarda cerca de 1,5 horas; se puede cerrar y volver a abrir (retoma donde quedo).
rem Resultados en out\review2\  (un JSON por corrida).
cd /d "%~dp0"
call venv_gpu\Scripts\activate
set SEED=42
echo ==== 1/5 INT8 corregido (EfficientNet-B0, MobileNetV2) ====
for %%M in (EfficientNet-B0 MobileNetV2) do python review2_cnn.py int8fix %%M
echo ==== 2/5 Conjunto geometrico, semilla 42 ====
for %%M in (EfficientNet-B0 MobileNetV2 EfficientNetV2-S ResNet50 MobileNetV3-Small) do python review2_cnn.py geo %%M
echo ==== 3/5 Receta ajustada, semilla 42 ====
for %%M in (EfficientNet-B0 MobileNetV2 MobileNetV3-Large) do python review2_cnn.py tuned %%M
echo ==== 4/5 Conjunto geometrico, semillas 1 y 2 ====
for %%S in (1 2) do (
  set SEED=%%S
  for %%M in (EfficientNet-B0 MobileNetV2 EfficientNetV2-S ResNet50 MobileNetV3-Small) do python review2_cnn.py geo %%M
)
echo ==== 5/5 Receta ajustada, semillas 1 y 2 ====
for %%S in (1 2) do (
  set SEED=%%S
  for %%M in (EfficientNet-B0 MobileNetV2) do python review2_cnn.py tuned %%M
)
set SEED=42
echo LISTO_REVIEW2_CNN
pause
