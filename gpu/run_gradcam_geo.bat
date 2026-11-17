@echo off
rem Observacion 1 del docente: Grad-CAM de la clase predicha y oclusion del tejido en el conjunto geometrico (337 imagenes).
rem Dos redes (EfficientNet-B0 y MobileNetV2), dos recetas (comun y ajustada) y las semillas 42, 1 y 2: 12 corridas.
rem Tarda cerca de 45 minutos; se puede cerrar y volver a abrir (retoma donde quedo).
rem Resultados en out\gradcam_geo\  (un .json y un .npz por corrida).
cd /d "%~dp0"
call venv_gpu\Scripts\activate
for %%S in (42 1 2) do (
  set SEED=%%S
  for %%M in (EfficientNet-B0 MobileNetV2) do (
    python gradcam_geo.py comun %%M
    python gradcam_geo.py ajustada %%M
  )
)
set SEED=42
echo LISTO_GRADCAM_GEO
pause
