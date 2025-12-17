# DeepSeek Desktop - IA Local

Aplicación de escritorio para ejecutar modelos DeepSeek localmente.
Proyecto para la materia de Inteligencia Artificial.

## Características

- ✅ Interfaz gráfica moderna y responsive
- ✅ Modelo DeepSeek local (sin internet)
- ✅ Streaming en tiempo real
- ✅ Guardado automático de conversaciones
- ✅ Ajuste de parámetros (temperatura, longitud)
- ✅ Tema claro/oscuro
- ✅ Historial de conversaciones
- ✅ Muy ligero (solo CPU necesario)

## Requisitos

- Python 3.8 o superior
- 4GB RAM mínimo
- 5GB espacio en disco
- No se requiere GPU

## Estructura

deepseek_desktop/
├── main.py                    # Aplicación principal 

├── deepseek_local.py          # Lógica OFFLINE del modelo

├── gui_app.py                 # Interfaz gráfica

├── download_model.py          # Descargador (solo con internet) 

├── start_offline.bat          # Iniciador offline (Windows) 

├── update_config_offline.py   # Configurador offline 

├── config.json               # Configuración

├── README.md                 # Documentación

├── models/                   # Modelos descargados 

├── data/                     # Datos y cache

(SE DEBE CREAR LAS CARPETA models y data MANUALMENTE)

## Instalación

PASOS A SEGUIR:

ABRIR LA TERMINAL Y EJECUTAR LO SIGUIENTE :D

------------- 1. Crear entorno virtual----------

python -m venv venv


----------- 2. Activar entorno virtual...----------

venv\Scripts\activate
 
--- 3. INSTALAR DEPENDENCIAS------

pip install torch transformers accelerate sentencepiece protobuf customtkinter pillow requests

---- 4. EJECUTAR ESTO (SOLO UNA VEZ) ------------

python download_model.py

----- 5. EJECUTAR ESTE ARCHIVO SIN INTERNET ---------

python main.py


## Sistema Offline Implementado

### 1. Caché Local
- Modelos guardados en `./models/`
- No requiere red después de primera descarga
- Verificación de archivos locales

### 2. Fallback Automático
- Si no hay modelo DeepSeek, usa GPT-2
- Siempre funciona, incluso sin descargas previas
- Ideal para demostraciones

### 3. Optimización
- Carga solo desde disco local
- No intenta conexiones HTTP
- Variables de entorno `TRANSFORMERS_OFFLINE=1`
