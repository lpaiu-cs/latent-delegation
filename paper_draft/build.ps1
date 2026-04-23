Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $paperDir
try {
    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
    if (-not $latexmk) {
        throw "latexmk is not installed. Install a TeX distribution with latexmk on PATH, then rerun this script."
    }

    & $latexmk.Source -pdf -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
