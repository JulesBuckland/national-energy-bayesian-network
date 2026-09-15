param(
    [switch]$nuts,
    [switch]$inla
)

$ErrorActionPreference = "Stop"

# Validate flags: exactly one of --nuts or --inla, default to --inla
if ($nuts -and $inla) { throw "Pass --nuts or --inla, not both." }
if (-not $nuts -and -not $inla) { $inla = $true }
$Engine = if ($nuts) { "nuts" } else { "inla" }

function Resolve-Python {
    <#
    .SYNOPSIS Returns the first working Python interpreter found.
    #>
    $candidates = @(
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            try { & $c --version 2>$null | Out-Null; return $c }
            catch {}
        }
    }
    return "python"
}

$py = Resolve-Python
Write-Host "=== National Pipeline (engine: $Engine, python: $py) ==="

function Invoke-Stage ([string]$Label, [string]$Module) {
    Write-Host "$Label..."
    & $py -m $Module
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
}

Invoke-Stage "Population synthesis" "src.data.population"
Invoke-Stage "GP emulator training" "src.inference.gp_emulator"

$inferModule = if ($Engine -eq "nuts") { "src.inference.model_unified" } else { "src.inference.inla.run_inla" }
Invoke-Stage "Bayesian inference ($Engine)" $inferModule

Write-Host "National run complete ($Engine engine)."
