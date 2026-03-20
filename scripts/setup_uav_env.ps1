param(
    [switch]$SkipSionna
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py37 = "D:\VS\SDK\Python37_64\python.exe"
$py312 = "D:\python\python.exe"
$venv = Join-Path $root ".venv"
$venvSionna = Join-Path $root ".venv_sionna"
$vendorDir = Join-Path $root "vendor_py37"
$baseReq = Join-Path $root "requirements\uav-base-py37.txt"
$sionnaReq = Join-Path $root "requirements\uav-sionna-py312.txt"

function Assert-Path($path, $label) {
    if (-not (Test-Path $path)) {
        throw "$label not found: $path"
    }
}

function Run-Checked($exe, [string[]]$cmdArgs) {
    & $exe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $exe $($cmdArgs -join ' ')"
    }
}

Assert-Path $py37 "Python 3.7"
Assert-Path $py312 "Python 3.12"
Assert-Path $baseReq "Base requirements"
Assert-Path $sionnaReq "Sionna requirements"

Write-Host "[1/4] Creating .venv with Python 3.7"
if (-not (Test-Path $venv)) {
    Run-Checked $py37 @("-m", "venv", $venv)
}

$venvPython = Join-Path $venv "Scripts\python.exe"

Write-Host "[2/4] Downloading Python 3.7 wheels via Python 3.12"
if (-not (Test-Path $vendorDir)) {
    New-Item -ItemType Directory -Path $vendorDir | Out-Null
}
Run-Checked $py312 @(
    "-m", "pip", "download",
    "--dest", $vendorDir,
    "--only-binary=:all:",
    "--platform", "win_amd64",
    "--implementation", "cp",
    "--python-version", "37",
    "--abi", "cp37m",
    "-r", $baseReq
)
Run-Checked $venvPython @("-m", "pip", "install", "--no-index", "--find-links", $vendorDir, "-r", $baseReq)

Write-Host "[3/4] Verifying main environment imports"
Run-Checked $venvPython @("-c", "import numpy, matplotlib, torch, geatpy, pypdf; print('main-env ok')")

if (-not $SkipSionna) {
    Write-Host "[4/4] Creating .venv_sionna with Python 3.12"
    if (-not (Test-Path $venvSionna)) {
        Run-Checked $py312 @("-m", "venv", $venvSionna)
    }

    $sionnaPython = Join-Path $venvSionna "Scripts\python.exe"
    Run-Checked $sionnaPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Run-Checked $sionnaPython @("-m", "pip", "install", "-r", $sionnaReq)
    Run-Checked $sionnaPython @("-m", "pip", "install", "torch==2.6.0", "pypdf")

    Write-Host "[4/4] Verifying Sionna environment imports"
    Run-Checked $sionnaPython @("-c", "import tensorflow as tf, sionna, torch; print(tf.__version__); print('sionna-env ok')")
} else {
    Write-Host "[4/4] Sionna setup skipped by request"
}

Write-Host "Environment setup completed."
