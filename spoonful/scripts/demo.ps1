# Windows version of demo.sh: preflight -> photon sidecar -> app (opens the browser; FSM homes the arm on boot)
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 [-Backend xarm|raw|sim]
param([string]$Backend = "xarm")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".venv\Scripts\python.exe"

& $py scripts\preflight.py --backend $Backend
if ($LASTEXITCODE -ne 0) { Write-Host "preflight failed"; exit $LASTEXITCODE }

$sidecar = $null
if ((Test-Path sidecar\node_modules) -and (Test-Path .env) -and (Select-String -Path .env -Pattern '^PHOTON_PROJECT_ID=.' -Quiet)) {
    $sidecar = Start-Process node -ArgumentList "photon.js" -WorkingDirectory sidecar -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput data\sidecar.log -RedirectStandardError data\sidecar.err.log
    Write-Host "photon sidecar started (data\sidecar.log)"
} else {
    Write-Host "photon sidecar skipped (npm install in sidecar\ and set PHOTON_* in .env to enable)"
}

try {
    & $py -m spoonful --backend $Backend run --open
} finally {
    if ($sidecar -and -not $sidecar.HasExited) { Stop-Process -Id $sidecar.Id -Force }
}
