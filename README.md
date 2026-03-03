# EEG-Library
Librería en Python para el procesamiento automatizado y análisis de señales EEG del casco Emotiv Epoc X, orientada al estudio de fatiga mental en contextos deportivos y académicos.
<br><br/>
<img width="1024" height="480" alt="Gemini_Generated_Image_7hm1xy7hm1xy7hm1" src="https://github.com/user-attachments/assets/fccc9847-7664-4eff-96a4-abebb1e61f63" />

---

## Requisitos previos

- Python **3.10 o superior**
- Git

---

## Instalación

### Windows (PowerShell)

```powershell
# 1. Clonar el repositorio
git clone https://github.com/Miguelit011/EEG-Library

# 2. Entrar a la carpeta del proyecto
cd EEG-Library

# 3. Crear el entorno virtual
python -m venv .venv

# 4. Activar el entorno virtual
.venv\Scripts\Activate.ps1

# 5. Actualizar pip y setuptools
pip install --upgrade pip setuptools

# 6. Instalar dependencias
pip install -e ".[dev]"
```

### Mac / Linux

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/EEG-Library.git

# 2. Entrar a la carpeta del proyecto
cd EEG-Library

# 3. Crear el entorno virtual
python3 -m venv .venv

# 4. Activar el entorno virtual
source .venv/bin/activate

# 5. Actualizar pip y setuptools
pip install --upgrade pip setuptools

# 6. Instalar dependencias
pip install -e ".[dev]"
```

---

## Uso básico

```python
from eeg_library import cargar_eeg

# Cargar un archivo EDF
raw = cargar_eeg("data/samples/sujeto_01.edf")

# raw es un objeto Raw de MNE con el que se puede seguir trabajando
print(raw.info)
```

Salida de ejemplo:

```
Archivo       : sujeto_01.edf
Frec. muestreo: 256.0 Hz
Duración      : 300.00 s
Canales (14)  : ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
```

---

## Comandos del día a día

Cada vez que retomes el proyecto, solo necesitas dos comandos:

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
python probar.py
```

**Mac / Linux:**
```bash
source .venv/bin/activate
python probar.py
```

---

## Estructura del proyecto

```
EEG-Library/
├── src/
│   └── eeg_library/
│       ├── __init__.py     # Punto de entrada del paquete
│       └── reader.py       # Carga y validación de archivos EDF
├── tests/                  # Tests automatizados con pytest
├── data/
│   └── samples/            # Archivos EDF de prueba (no se suben al repo)
├── pyproject.toml          # Configuración del proyecto y dependencias
└── README.md
```
