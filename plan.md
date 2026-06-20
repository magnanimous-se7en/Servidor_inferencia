# Plan: Servidor de Inferencia para Detección de Cáncer en Tomografías

## Contexto
Necesito crear un servidor de inferencia HTTP que cargue un modelo ONNX de detección de cáncer en tomografías y exponga un endpoint REST para que un backend externo pueda enviarle imágenes y recibir predicciones. El deploy final será en Render.com (plan gratuito, solo CPU).

---

## Tarea

Crea el proyecto completo `inference-server/` con la siguiente estructura y archivos:

```
inference-server/
├── model/
│   └── .gitkeep
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Archivos a crear

### `main.py`
Servidor FastAPI con las siguientes características:
- Cargar el modelo ONNX desde `model/modelo.onnx` al iniciar con `onnxruntime`
- Middleware CORS abierto (`allow_origins=["*"]`) para que el backend externo pueda conectarse
- Endpoint `GET /` que retorna `{"status": "ok", "message": "Inference server running"}`
- Endpoint `POST /predict` que:
  - Recibe un archivo de imagen via `multipart/form-data` (campo `file`)
  - Preprocesa la imagen: convertir a RGB, redimensionar a 224x224, normalizar dividiendo entre 255.0, transponer de HWC a CHW, añadir dimensión de batch
  - Corre inferencia con `onnxruntime`
  - Retorna JSON con los campos: `has_cancer` (bool), `confidence` (float redondeado a 4 decimales), `label` (string: "cancer" o "normal")
- Manejo de errores con `HTTPException` para imágenes inválidas o error en inferencia
- Comentarios en español explicando cada sección importante

### `requirements.txt`
Incluir exactamente estas dependencias sin versiones fijas:
```
fastapi
uvicorn
onnxruntime
pillow
numpy
python-multipart
```

### `.gitignore`
Ignorar:
- `__pycache__/` y archivos `*.pyc`
- Archivos `.onnx` (pueden pesar más de 100MB)
- Entornos virtuales: `venv/`, `env/`, `.venv/`
- `.env`
- `*.egg-info/`
- `dist/`, `build/`

### `README.md`
Documentación en español con las siguientes secciones:
1. **Descripción** — qué hace el servidor
2. **Requisitos** — Python 3.9+, pip
3. **Instalación** — pasos para instalar dependencias y colocar el modelo
4. **Uso local** — cómo correr con `uvicorn main:app --reload` y acceder a `/docs`
5. **Endpoints** — tabla con `GET /` y `POST /predict`, sus parámetros y respuesta esperada
6. **Deploy en Render.com** — pasos resumidos: conectar repo GitHub, build command, start command (`uvicorn main:app --host 0.0.0.0 --port $PORT`)
7. **Notas importantes** — advertencia de que el preprocessing (tamaño de imagen y normalización) debe confirmarse con quien entrenó el modelo

### `model/.gitkeep`
Archivo vacío para que Git trackee la carpeta `model/` aunque no tenga el `.onnx`.

---

## Entorno virtual e instalación

Después de crear todos los archivos, ejecutar en terminal los siguientes comandos en orden:

1. Crear el entorno virtual:
   ```bash
   python -m venv venv
   ```

2. Activarlo:
   - En Windows: `venv\Scripts\activate`
   - En Mac/Linux: `source venv/bin/activate`

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Verificar que el servidor levanta correctamente:
   ```bash
   uvicorn main:app --reload
   ```

Si algo falla en la instalación, mostrar el error y sugerir solución.

---

## Instrucciones adicionales

- Todo el código debe estar en **español** en comentarios y docstrings
- Usar **type hints** en todas las funciones de Python
- El servidor debe poder iniciar aunque no exista el archivo `model/modelo.onnx`, mostrando un warning claro en consola en lugar de crashear al importar — solo fallar en el momento de recibir una request a `/predict`
- Después de crear todos los archivos, mostrar el comando para instalar dependencias y levantar el servidor localmente
