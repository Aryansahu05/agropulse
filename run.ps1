Param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir "venv\Scripts\python.exe"
$appPath = Join-Path $scriptDir "app\app.py"
$reqPath = Join-Path $scriptDir "requirements.txt"

function Resolve-SystemPython {
    # Prefer the Python launcher when available (lets us pick 3.12 explicitly).
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        # Prefer 3.12 (best compatibility today), but fall back gracefully.
        try { & py -3.12 -c "import sys; print(sys.executable)" 2>$null | Out-Null; return @{ kind = "py"; args = @("-3.12") } } catch {}
        try { & py -3.13 -c "import sys; print(sys.executable)" 2>$null | Out-Null; return @{ kind = "py"; args = @("-3.13") } } catch {}
        return @{ kind = "py"; args = @() }
    }

    # Fallback to python on PATH.
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        return @{ kind = "python"; args = @() }
    }

    return $null
}

function Ensure-Venv {
    if (Test-Path $venvPython) { return }
    Write-Host "Creating virtual environment in .\venv ..."
    $sysPy = Resolve-SystemPython
    if ($null -eq $sysPy) {
        throw "Python not found. Install Python 3.12 and ensure 'py' or 'python' is available in PATH."
    }

    if ($sysPy.kind -eq "py") {
        & py @($sysPy.args) -m venv (Join-Path $scriptDir "venv")
    } else {
        & python -m venv (Join-Path $scriptDir "venv")
    }
    if (-not (Test-Path $venvPython)) {
        throw "Failed to create venv at $venvPython"
    }
}

if (-not (Test-Path $appPath)) {
    Write-Host "Application file not found at $appPath"
    exit 1
}

if (-not (Test-Path $reqPath)) {
    Write-Host "requirements.txt not found at $reqPath"
    exit 1
}

Ensure-Venv

if (-not (Test-Path $venvPython)) {
    throw "Venv Python not found at $venvPython. Delete the 'venv' folder and rerun, or verify Python installation."
}

Write-Host "Installing/updating dependencies..."
& $venvPython -m pip install --upgrade pip | Out-Host
& $venvPython -m pip install -r $reqPath | Out-Host

Write-Host "Starting AgroPulse Flask app..."
& $venvPython $appPath

