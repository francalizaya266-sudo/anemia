@echo off
rem Tercera ronda: receta ajustada en el conjunto geometrico (337 imagenes) y semillas 1 y 2 de MobileNetV3-Large.
rem Tarda cerca de 50 minutos; se puede cerrar y volver a abrir (retoma donde quedo).
rem Resultados en out\review2\  (un JSON por corrida: cnn_*_tunedgeo_s*.json, cnn_MobileNetV3-Large_tuned_s1/s2.json).
cd /d "%~dp0"
call venv_gpu\Scripts\activate
for %%S in (42 1 2) do (
  set SEED=%%S
  for %%M in (EfficientNet-B0 MobileNetV2 MobileNetV3-Large) do python review2_cnn.py tunedgeo %%M
)
for %%S in (1 2) do (
  set SEED=%%S
  python review2_cnn.py tuned MobileNetV3-Large
)
set SEED=42
echo LISTO_REVIEW3_CNN
pause
