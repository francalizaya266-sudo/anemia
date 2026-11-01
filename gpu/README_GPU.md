# Entrenamiento de CNN en la RTX 4060 (Windows, PyTorch + CUDA)

Carpeta autocontenida. No toca el resto del proyecto.

## 1. Preparar (una sola vez, ~5 min)

```bat
cd "C:\AnemIA-E2\Nueva carpeta"
tar -xf cp-anemic.zip          :: crea cp-anemic\images y cp-anemic\metadata.csv
py -3.11 -m venv venv_gpu      :: 3.10 a 3.12 sirven; 3.13 tambien si torch ya lo soporta
venv_gpu\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_gpu.txt
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Debe imprimir `True NVIDIA GeForce RTX 4060 ...`. Si imprime `False`, actualizar el driver NVIDIA (>= 550) y repetir.

## 2. Correr todo

```bat
run_all.bat
```

Por arquitectura y protocolo tarda 2 a 5 min en la 4060. Resultados en `out\`:

- `cnn_<modelo>_clean.json`: metricas con IC 95 % para FP32, FP16, INT8 dinamico e INT8 estatico, mas tamano (MB) y latencia (ms, 1 hilo CPU, onnxruntime).
- `cnn_<modelo>_naive.json`: protocolo ingenuo (708 filas con duplicados) para cuantificar la fuga.
- `<modelo>_*.onnx`: modelos exportados (FP32/FP16/INT8). Para Android se usan con ONNX Runtime Mobile o se convierten a TFLite con `onnx2tf`.
- `gradcam_<modelo>.png/.json`: explicabilidad.

## 3. Que hace cada script

- `common.py`: carga, deduplicacion (471 imagenes unicas), pliegues, metricas, bootstrap.
- `train_torch.py`: entrenamiento en 2 fases (cabeza, luego ajuste fino completo con BatchNorm congelada), aumento solo geometrico, CV 5 pliegues, exportacion y medicion.
- `gradcam_torch.py`: Grad-CAM + fraccion de atencion dentro de la conjuntiva.
- `meta_groups.csv`: metadata original + columna `grp` (grupo de duplicados por SHA-256 y pHash).

Variables utiles: `set NFOLDS=1` para una prueba rapida; `set DATA_DIR=C:\ruta\cp-anemic` si el dataset esta en otro lado.
