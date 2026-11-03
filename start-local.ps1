$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment is missing: $python"
}
$expectedVersion = & $python -c "import sys; sys.path.insert(0, sys.argv[1]); from app import __version__; print(__version__)" $projectRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to read the project version. Check the Python environment.'
}

$health = $null
try {
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
} catch {
    # Start the server below when nothing is listening on the local endpoint.
}
if ($health) {
    if ($health.status -eq 'ok' -and $health.version -eq $expectedVersion -and $health.dsh_runtime -eq 'native') {
        Write-Host 'Already running at http://127.0.0.1:8000'
        return
    }
    throw 'Port 8000 has a different pipeline server. Stop that server before starting this one.'
}

Push-Location $projectRoot
try {
    & $python -m app.main
} finally {
    Pop-Location
}
