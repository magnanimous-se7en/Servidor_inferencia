# API — Servidor de Inferencia (Detección de Cáncer en Tomografías)

Referencia de la API para integrar el servidor de inferencia desde el backend.

El servicio recibe una imagen de tomografía y clasifica el tejido en una de
cuatro categorías: HGC, LGC, NST o NTL.

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

Endpoint principal. Recibe una imagen y devuelve la clasificación del modelo.

- **Content-Type:** `multipart/form-data`
- **Campo requerido:** `file` → archivo de imagen (PNG, JPG, etc.).

Se envía la imagen **tal cual**; el servidor la redimensiona a 224×224 y la
normaliza internamente. **No** hay que preprocesar nada del lado del backend.

**Respuesta `200` (éxito):**

```json
{
  "has_cancer": true,
  "confidence": 0.8731,
  "label": "HGC"
}
```

| Campo        | Tipo     | Descripción                                                              |
| ------------ | -------- | ------------------------------------------------------------------------ |
| `has_cancer` | `bool`   | `true` cuando `label` es `"HGC"` o `"LGC"` (tejido cancerígeno)        |
| `confidence` | `float`  | Probabilidad de la clase predicha, redondeada a 4 decimales             |
| `label`      | `string` | Clase predicha: `"HGC"`, `"LGC"`, `"NST"` o `"NTL"` (ver tabla abajo) |

**Significado de las clases:**

| Clase | Significado        | `has_cancer` |
| ----- | ------------------ | ------------ |
| `HGC` | High Grade Cancer  | `true`       |
| `LGC` | Low Grade Cancer   | `true`       |
| `NST` | Normal Surrounding Tissue | `false` |
| `NTL` | Normal Tissue-Like | `false`      |

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

const data = await res.json();
// { has_cancer: true, confidence: 0.8731, label: "HGC" }
console.log(data.label, data.has_cancer ? "CANCER" : "NORMAL");
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
data = res.json()
# { 'has_cancer': True, 'confidence': 0.8731, 'label': 'HGC' }
print(data["label"], "CANCER" if data["has_cancer"] else "NORMAL")
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

## Nota sobre el preprocesamiento

El servidor aplica: conversión a RGB, redimensionado a **224×224**, normalización
dividiendo entre **255.0**, orden de canales **CHW** y dimensión de batch
`(1, 3, 224, 224)`. La salida del modelo (4 logits) se pasa por softmax para
obtener probabilidades; la clase con mayor probabilidad es la predicción final.
