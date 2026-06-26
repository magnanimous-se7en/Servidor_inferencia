"""
Convierte best_model_resnet50.pth.zip (PyTorch) a model/modelo.onnx (ONNX).

Uso:
    python convert_to_onnx.py

Arquitectura detectada:
    backbone = ResNet50 con fc personalizado:
        Sequential(Dropout, Linear(2048→256), BatchNorm1d(256), ReLU, Dropout, Linear(256→4))
    Clases (índice 0-3): HGC, LGC, NST, NTL
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

RUTA_PTH_ZIP = "../best_model_resnet50.pth.zip"
RUTA_ONNX    = "model/modelo.onnx"

# --------------------------------------------------------------------------- #
# 1. Reconstruir la arquitectura exacta del modelo entrenado
# --------------------------------------------------------------------------- #

class ModeloCancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 4),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# --------------------------------------------------------------------------- #
# 2. Cargar los pesos desde el .pth.zip
# --------------------------------------------------------------------------- #

print(f"[INFO] Cargando pesos desde '{RUTA_PTH_ZIP}' ...")
state_dict = torch.load(RUTA_PTH_ZIP, map_location="cpu", weights_only=False)

if not isinstance(state_dict, dict) or not all(
    isinstance(v, torch.Tensor) for v in state_dict.values()
):
    print("[ERROR] El checkpoint no tiene el formato esperado (state_dict plano).")
    print("Claves encontradas:", list(state_dict.keys())[:5])
    sys.exit(1)

model = ModeloCancer()
model.load_state_dict(state_dict, strict=True)
model.eval()
print("[OK] Pesos cargados correctamente.")

# --------------------------------------------------------------------------- #
# 3. Exportar a ONNX autocontenido (sin archivo .data externo)
# --------------------------------------------------------------------------- #

print(f"[INFO] Exportando a ONNX -> '{RUTA_ONNX}' ...")
dummy = torch.zeros(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    RUTA_ONNX,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)

size_mb = os.path.getsize(RUTA_ONNX) / 1024 / 1024
print(f"[OK] Exportado. Tamanio: {size_mb:.1f} MB -> {os.path.abspath(RUTA_ONNX)}")
