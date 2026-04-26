# LUT-KAN v15 — Windows PowerShell runner
# Usage from project root:
#   .\run.ps1 experiment   — run H8 corrected experiment
#   .\run.ps1 test         — run test suite
#   .\run.ps1 quick        — quick smoke test (30 sec)

param(
    [string]$Task = "quick"
)

$env:PYTHONPATH = "src"
$python = "python"   # change to "python3" if needed

switch ($Task) {
    "test" {
        Write-Host "Running test suite..." -ForegroundColor Cyan
        & $python -m pytest tests/test_stack.py -v
    }
    "quick" {
        Write-Host "Quick smoke test..." -ForegroundColor Cyan
        & $python scripts/run_quick.py
    }
    "experiment" {
        Write-Host "Running H8 corrected experiment (this takes ~30 min)..." -ForegroundColor Cyan
        & $python scripts/run_h8_corrected.py
    }
    "coverage" {
        Write-Host "Coverage diagnostics on trained model..." -ForegroundColor Cyan
        & $python scripts/run_coverage.py
    }
    default {
        Write-Host "Unknown task: $Task" -ForegroundColor Red
        Write-Host "Available: test, quick, experiment, coverage"
    }
}
