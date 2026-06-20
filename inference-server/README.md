# Inference Server — Detección de Cáncer en Tomografías

## Descripción

Servidor HTTP de inferencia construido con **FastAPI** que carga un modelo
**ONNX** de detección de cáncer en tomografías y expone un endpoint REST
(`POST /predict`). Un backend externo puede enviarle una imagen vía
`multipart/form-data` y recibir la predicción en formato JSON.

El servidor está optimizado para correr **solo en CPU**, pensado para deploy en
**Render.com** (plan gratuito).

---

## Requisitos

- **Python 3.9+**
- **pip**

---

## Instalación

1. Clona el repositorio y entra a la carpeta del proyecto:

   ```bash
   cd inference-server
   ```

2. Crea y activa un entorno virtual (ver sección [Uso local](#uso-local)).

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Coloca tu modelo ONNX dentro de la carpeta `model/` con el nombre exacto
   **`modelo.onnx`**:

   ```
   model/modelo.onnx
   ```

   > El archivo `.onnx` está ignorado en `.gitignore` porque suele pesar más de
   > 100MB. Debes copiarlo manualmente en cada entorno.

---

## Uso local

1. Crea el entorno virtual:

   ```bash
   python -m venv venv
   ```

2. Actívalo:

   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Levanta el servidor en modo desarrollo:

   ```bash
   uvicorn main:app --reload
   ```

5. Abre la documentación interactiva (Swagger UI) en tu navegador:

   ```
   http://127.0.0.1:8000/docs
   ```

   Desde ahí puedes probar el endpoint `/predict` subiendo una imagen.

---

## Endpoints

| Método | Ruta       | Parámetros                                   | Respuesta esperada                                                                 |
| ------ | ---------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| `GET`  | `/`        | —                                            | `{"status": "ok", "message": "Inference server running"}`                         |
| `POST` | `/predict` | `file`: imagen (`multipart/form-data`)       | `{"has_cancer": bool, "confidence": float, "label": "cancer" \| "normal"}`        |

### Ejemplo de respuesta de `/predict`

```json
{
  "has_cancer": true,
  "confidence": 0.9123,
  "label": "cancer"
}
```

### Ejemplo de llamada con `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@ruta/a/tomografia.png"
```

---

## Deploy en Render.com

1. Sube el proyecto a un repositorio de **GitHub**.
2. En Render.com crea un nuevo **Web Service** y conecta el repositorio.
3. Configura los comandos:

   - **Build command:**

     ```bash
     pip install -r requirements.txt
     ```

   - **Start command:**

     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```

4. Asegúrate de que el modelo `modelo.onnx` esté disponible en el entorno.
   Como el `.onnx` no se versiona en Git, puedes:
   - Subirlo mediante un disco persistente de Render, o
   - Descargarlo en el build desde un almacenamiento externo (S3, Drive, etc.).

---

## Notas importantes

> ⚠️ **El preprocesamiento debe confirmarse con quien entrenó el modelo.**
>
> Este servidor asume un preprocesamiento estándar:
> - Conversión a **RGB**
> - Redimensionado a **224x224**
> - Normalización dividiendo entre **255.0** (rango `[0, 1]`)
> - Orden de canales **CHW** con dimensión de batch `(1, 3, 224, 224)`
>
> Si el modelo fue entrenado con otro tamaño de entrada, otra normalización
> (por ejemplo media/desviación de ImageNet) u otro orden de canales, las
> predicciones serán incorrectas. **Confirma estos valores antes de usar el
> servidor en producción.**
>
> Del mismo modo, la interpretación de la salida (sigmoide vs. softmax y el
> índice de la clase "cancer") debe validarse contra el modelo real.
