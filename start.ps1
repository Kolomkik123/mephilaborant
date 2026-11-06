param([string]$Command = 'ui')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
# The portable interpreter also uses the existing .venv packages.
$portablePython = Join-Path $projectRoot '.runtime\python.exe'
if (Test-Path -LiteralPath $portablePython) {
    & $portablePython -X utf8 (Join-Path $projectRoot 'run.py') $Command
    exit $LASTEXITCODE
}
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        & $venvPython -X utf8 (Join-Path $projectRoot 'run.py') $Command
        exit $LASTEXITCODE
    }
}
throw 'No working interpreter. Restore .runtime or create .venv with Python 3.12 and pip install -r requirements.txt.'
