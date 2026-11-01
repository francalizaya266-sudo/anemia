@echo off
cd /d "%~dp0"
call venv_gpu\Scripts\activate
set SEED=42
echo ==== Grad-CAM predicted class + occlusion (5 folds) ====
python gradcam_review2.py
echo LISTO_GRADCAM2
