"""
Servidor de inferencia para detección de cáncer en tomografías.

Carga un modelo ONNX al iniciar y expone un endpoint REST (`POST /predict`)
para que un backend externo envíe imágenes y reciba predicciones.

El servidor está pensado para correr en CPU (Render.com plan gratuito).
"""

from io import BytesIO
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

# Ruta al modelo ONNX. Puede no existir al iniciar (ver carga diferida abajo).
RUTA_MODELO: str = "model/modelo.onnx"

# Tamaño al que se redimensiona la imagen antes de la inferencia.
# IMPORTANTE: confirmar con quien entrenó el modelo (ver README -> Notas).
TAMANO_ENTRADA: tuple[int, int] = (224, 224)

# Umbral de decisión: si la confianza supera este valor se considera "cancer".
UMBRAL_DECISION: float = 0.5

# --------------------------------------------------------------------------- #
# Aplicación FastAPI
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Inference Server - Detección de cáncer en tomografías",
    description="Servidor HTTP que corre un modelo ONNX sobre imágenes de tomografía.",
    version="1.0.0",
)

# Middleware CORS abierto para que el backend externo pueda conectarse desde
# cualquier origen. En producción conviene restringir `allow_origins`.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Carga del modelo (diferida y tolerante a fallos)
# --------------------------------------------------------------------------- #

# Mantenemos la sesión de ONNX en una variable global. Si el modelo no existe
# al iniciar, NO crasheamos: dejamos la sesión en None y mostramos un warning.
# El error solo se lanzará cuando llegue una request a /predict.
sesion_onnx: Optional[ort.InferenceSession] = None


def cargar_modelo() -> None:
    """Intenta cargar el modelo ONNX en la sesión global.

    No lanza excepción si el archivo no existe: solo registra un warning en
    consola para permitir que el servidor arranque igualmente.
    """
    global sesion_onnx
    try:
        sesion_onnx = ort.InferenceSession(
            RUTA_MODELO,
            providers=["CPUExecutionProvider"],
        )
        print(f"[OK] Modelo ONNX cargado correctamente desde '{RUTA_MODELO}'.")
    except Exception as error:  # noqa: BLE001 - queremos capturar cualquier fallo de carga
        sesion_onnx = None
        print(
            f"[WARNING] No se pudo cargar el modelo desde '{RUTA_MODELO}': {error}\n"
            "          El servidor arrancará igualmente, pero las requests a "
            "/predict fallarán hasta que el modelo esté disponible."
        )


@app.on_event("startup")
def evento_startup() -> None:
    """Carga el modelo cuando el servidor inicia."""
    cargar_modelo()


# --------------------------------------------------------------------------- #
# Preprocesamiento de la imagen
# --------------------------------------------------------------------------- #

def preprocesar_imagen(datos: bytes) -> np.ndarray:
    """Convierte los bytes de una imagen en el tensor de entrada del modelo.

    Pasos:
      1. Abrir la imagen y convertir a RGB.
      2. Redimensionar a `TAMANO_ENTRADA` (224x224 por defecto).
      3. Normalizar dividiendo entre 255.0 (rango [0, 1]).
      4. Transponer de HWC (alto, ancho, canal) a CHW (canal, alto, ancho).
      5. Añadir la dimensión de batch -> (1, 3, 224, 224).

    Args:
        datos: contenido binario del archivo de imagen recibido.

    Returns:
        np.ndarray de tipo float32 listo para la inferencia.

    Raises:
        ValueError: si los bytes no corresponden a una imagen válida.
    """
    try:
        imagen = Image.open(BytesIO(datos)).convert("RGB")
    except Exception as error:  # noqa: BLE001 - PIL lanza distintos tipos de error
        raise ValueError(f"La imagen no es válida o está corrupta: {error}") from error

    # Redimensionar a la resolución esperada por el modelo.
    imagen = imagen.resize(TAMANO_ENTRADA)

    # A array float32 y normalización a [0, 1].
    arreglo = np.asarray(imagen, dtype=np.float32) / 255.0

    # HWC -> CHW.
    arreglo = np.transpose(arreglo, (2, 0, 1))

    # Añadir dimensión de batch -> (1, 3, H, W).
    arreglo = np.expand_dims(arreglo, axis=0)

    return arreglo


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def raiz() -> dict[str, str]:
    """Healthcheck simple para verificar que el servidor está vivo."""
    return {"status": "ok", "message": "Inference server running"}


@app.post("/predict")
async def predecir(file: UploadFile = File(...)) -> dict[str, object]:
    """Recibe una imagen y devuelve la predicción del modelo.

    Espera un archivo de imagen vía `multipart/form-data` en el campo `file`.

    Returns:
        dict con:
          - has_cancer (bool): True si se detecta cáncer.
          - confidence (float): confianza redondeada a 4 decimales.
          - label (str): "cancer" o "normal".
    """
    # 1. Verificar que el modelo esté disponible.
    if sesion_onnx is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"El modelo no está cargado. Coloca el archivo en '{RUTA_MODELO}' "
                "y reinicia el servidor."
            ),
        )

    # 2. Leer y preprocesar la imagen.
    try:
        contenido = await file.read()
        tensor_entrada = preprocesar_imagen(contenido)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    # 3. Correr la inferencia con onnxruntime.
    try:
        nombre_entrada = sesion_onnx.get_inputs()[0].name
        salidas = sesion_onnx.run(None, {nombre_entrada: tensor_entrada})
    except Exception as error:  # noqa: BLE001 - cualquier fallo de inferencia
        raise HTTPException(
            status_code=500,
            detail=f"Error al ejecutar la inferencia: {error}",
        ) from error

    # 4. Interpretar la salida del modelo.
    #    Se asume que la primera salida contiene la(s) probabilidad(es).
    #    Soportamos tanto salida escalar (sigmoide) como vector (softmax).
    try:
        prediccion = np.asarray(salidas[0]).flatten()

        if prediccion.size == 1:
            # Salida tipo sigmoide: una sola probabilidad de "cancer".
            confianza = float(prediccion[0])
        else:
            # Salida tipo softmax: tomamos la probabilidad de la clase "cancer".
            # Convención asumida: índice 1 = cancer, índice 0 = normal.
            confianza = float(prediccion[1])

        tiene_cancer = confianza >= UMBRAL_DECISION
        etiqueta = "cancer" if tiene_cancer else "normal"
    except Exception as error:  # noqa: BLE001 - salida con formato inesperado
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo interpretar la salida del modelo: {error}",
        ) from error

    return {
        "has_cancer": tiene_cancer,
        "confidence": round(confianza, 4),
        "label": etiqueta,
    }
