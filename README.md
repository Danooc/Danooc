# Danooc

Red neuronal convolucional (CNN) para clasificación de imágenes, construida con **PyTorch**.

El modelo — **DanoocNet** — se entrena sobre el dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) y aprende a reconocer 10 categorías:

> airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

---

## Estructura del proyecto

```
danooc/
├── __init__.py        # Exportaciones del paquete
├── config.py          # Hiperparámetros y configuración
├── model.py           # Arquitectura de DanoocNet (CNN)
├── dataset.py         # Carga de CIFAR-10 con augmentación
├── trainer.py         # Bucle de entrenamiento y evaluación
└── predict.py         # Inferencia sobre una imagen individual
scripts/
├── train.py           # Script para entrenar el modelo
├── evaluate.py        # Script para evaluar un checkpoint
└── predict.py         # Script para clasificar una imagen
```

## Requisitos

- Python 3.10+
- PyTorch 2.0+

```bash
pip install -r requirements.txt
```

## Entrenamiento

Entrena el modelo con los parámetros por defecto (20 épocas, batch size 64):

```bash
python scripts/train.py
```

Personaliza los hiperparámetros:

```bash
python scripts/train.py --epochs 30 --batch-size 128 --lr 0.0005
```

El dataset CIFAR-10 se descarga automáticamente en `./data/`. Los checkpoints se guardan en `./checkpoints/`.

## Evaluación

Evalúa el mejor modelo guardado sobre el set de pruebas:

```bash
python scripts/evaluate.py
python scripts/evaluate.py --checkpoint last.pt
```

## Predicción

Clasifica cualquier imagen:

```bash
python scripts/predict.py ruta/a/imagen.png
```

## Arquitectura del modelo

DanoocNet usa tres bloques convolucionales seguidos de un clasificador fully-connected:

| Capa | Salida |
|---|---|
| Conv2d(3→32) + BN + ReLU + MaxPool | 16×16×32 |
| Conv2d(32→64) + BN + ReLU + MaxPool | 8×8×64 |
| Conv2d(64→128) + BN + ReLU + MaxPool | 4×4×128 |
| Flatten + Linear(2048→256) + ReLU + Dropout | 256 |
| Linear(256→10) | 10 |

Total de parámetros: ~581K

## Uso como librería

```python
from danooc import DanoocNet, get_dataloaders, Trainer
from danooc.config import Config

config = Config(epochs=10, learning_rate=0.001)
train_loader, test_loader = get_dataloaders(batch_size=config.batch_size)

model = DanoocNet(num_classes=10)
trainer = Trainer(model, config)
history = trainer.train(train_loader, test_loader)
```

## Licencia

MIT
