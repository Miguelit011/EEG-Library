# Fatiga aguda — deep learning, paso 4, fase D: evaluación única en test.
#
# Lanza seguidos los tres pasos de test (preparar, puntuar y evaluar) y para
# si alguno falla. Solo se ejecuta UNA vez, con el preregistro, los scripts y
# configuracion_dl.json ya commiteados (los scripts comprueban que git está
# limpio). Cada script escribe su propio registro en la carpeta de salida de
# EEG-Data/processed/; no redirigir la salida.
#
# Ejecutar desde cualquier carpeta:
#     powershell -ExecutionPolicy Bypass -File analisis\fatiga_aguda\deep_learning\04_test_eegnet\ejecutar_test.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$raiz = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $raiz
$carpeta = "analisis\fatiga_aguda\deep_learning\04_test_eegnet"
$venv = Join-Path $raiz ".venv\Scripts\python.exe"
$dl = "C:\venvs\eeg-dl\Scripts\python.exe"

& $venv "$carpeta\preparar.py" --test --permitir-test
if ($LASTEXITCODE -ne 0) { throw "preparar.py --test falló (código $LASTEXITCODE)" }
& $dl "$carpeta\entrenar.py" --puntuar-test --permitir-test
if ($LASTEXITCODE -ne 0) { throw "entrenar.py --puntuar-test falló (código $LASTEXITCODE)" }
& $venv "$carpeta\evaluar.py" --test --permitir-test
if ($LASTEXITCODE -ne 0) { throw "evaluar.py --test falló (código $LASTEXITCODE)" }
Write-Output "Evaluación única en test terminada."
