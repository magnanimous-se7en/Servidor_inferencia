# API — Servidor de Inferencia (Detección de Cáncer en Tomografías)

Referencia de la API para integrar el servidor de inferencia desde el backend.

El servicio recibe una imagen de tomografía y devuelve si detecta cáncer junto
con la confianza del modelo.

---

## Base URL

| Entorno | URL |
| ------- | --- |
| **Producción (Render)** | `https://servidor-inferencia.onrender.com` |
| Local | `http://127.0.0.1:8000` |

> **CORS abierto** (`Access-Control-Allow-Origin: *`): el backend puede llamar a
> la API desde cualquier origen sin configuración adicional.

---

## Endpoints

### `GET /` — Healthcheck

Verifica que el servidor esté vivo (útil para monitoreo).

**Respuesta `200`:**

```json
{ "status": "ok", "message": "Inference server running" }
```

---

### `POST /predict` — Predicción

Endpoint principal. Recibe una imagen y devuelve la predicción del modelo.

- **Content-Type:** `multipart/form-data`
- **Campo requerido:** `file` → archivo de imagen (PNG, JPG, etc.).

Se envía la imagen **tal cual**; el servidor la redimensiona a 224×224 y la
normaliza internamente. **No** hay que preprocesar nada del lado del backend.

**Respuesta `200` (éxito):**

```json
{
  "has_cancer": true,
  "confidence": 0.9123,
  "label": "cancer"
}
```

| Campo        | Tipo     | Descripción                                       |
| ------------ | -------- | ------------------------------------------------- |
| `has_cancer` | `bool`   | `true` si la confianza ≥ 0.5                       |
| `confidence` | `float`  | Probabilidad de cáncer, redondeada a 4 decimales  |
| `label`      | `string` | `"cancer"` o `"normal"`                            |

---

## Errores

Todos los errores devuelven un cuerpo con el formato `{ "detail": "<mensaje>" }`.

| Código | Significado                                                        |
| ------ | ----------------------------------------------------------------- |
| `400`  | La imagen es inválida o está corrupta                             |
| `503`  | El modelo ONNX no está cargado en el servidor                     |
| `500`  | Error al ejecutar la inferencia o al interpretar la salida        |

---

## Ejemplos de consumo

### Node.js (18+) — `fetch` + `FormData`

```js
const fd = new FormData();
fd.append("file", fileBlob, "tomografia.png"); // Blob/File (o Buffer en Node)

const res = await fetch("https://servidor-inferencia.onrender.com/predict", {
  method: "POST",
  body: fd,
});

if (!res.ok) {
  const { detail } = await res.json();
  throw new Error(`Error ${res.status}: ${detail}`);
}

const data = await res.json(); // { has_cancer, confidence, label }
console.log(data);
```

### Python — `requests`

```python
import requests

with open("tomografia.png", "rb") as f:
    res = requests.post(
        "https://servidor-inferencia.onrender.com/predict",
        files={"file": f},
    )

res.raise_for_status()
print(res.json())  # { 'has_cancer': ..., 'confidence': ..., 'label': ... }
```

### curl

```bash
curl -X POST "https://servidor-inferencia.onrender.com/predict" \
  -F "file=@tomografia.png"
```

---

## Documentación auto-generada

FastAPI expone la especificación automáticamente:

- **Swagger UI (interactivo):** <https://servidor-inferencia.onrender.com/docs>
  — permite probar `/predict` subiendo imágenes desde el navegador.
- **OpenAPI JSON:** <https://servidor-inferencia.onrender.com/openapi.json>
  — importable en Postman / Insomnia o para generar un cliente automáticamente.

---

## ⚠️ Estado actual / limitación conocida

A la fecha, **`POST /predict` devuelve `503 "El modelo no está cargado"`** en
producción. Esto ocurre porque el archivo `model/modelo.onnx` está ignorado en
git (pesa más de 100MB) y aún **no está disponible en el servidor de Render**.

- El servidor está online (`GET /` responde `200`).
- El **contrato de la API NO cambiará** cuando el modelo esté disponible: una vez
  que el `.onnx` esté en Render, `/predict` empezará a devolver el JSON de
  predicción descrito arriba.
- El backend ya puede integrarse contra este contrato; conviene manejar el `503`
  de forma controlada mientras tanto.

---

## Nota sobre el preprocesamiento

El servidor asume el siguiente preprocesamiento: conversión a RGB, redimensionado
a **224×224**, normalización dividiendo entre **255.0**, orden de canales **CHW**
y dimensión de batch `(1, 3, 224, 224)`. La interpretación de la salida asume
sigmoide (una probabilidad) o softmax (índice 1 = "cancer").

> Estos valores deben **confirmarse con quien entrenó el modelo**. Si el modelo
> se entrenó con otro tamaño, otra normalización u otro orden de clases, las
> predicciones serán incorrectas (ver `inference-server/README.md`).
