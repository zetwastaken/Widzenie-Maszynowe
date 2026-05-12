$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Brak .venv. Utworz srodowisko i zainstaluj zaleznosci: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

$ModelPath = Join-Path $ProjectRoot "face_landmarker.task"
if (-not (Test-Path $ModelPath)) {
    throw "Brak pliku face_landmarker.task w katalogu projektu."
}

& $Python -m pip install -r requirements-build.txt
& $Python -m PyInstaller --noconfirm --clean BestFrameAnalyzer.spec

$OutputDir = Join-Path $ProjectRoot "dist\BestFrameAnalyzer"
$ExePath = Join-Path $OutputDir "BestFrameAnalyzer.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build nie utworzyl pliku: $ExePath"
}

Write-Host ""
Write-Host "Gotowe: $ExePath"
