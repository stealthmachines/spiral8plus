# Environment Check Script for Grand Unified Theory Testing
# ========================================================

$ErrorActionPreference = "Continue"

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "GUT Testing Environment Check" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

$allGood = $true

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  OK: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found" -ForegroundColor Red
    $allGood = $false
}

# Check Docker
Write-Host "`nChecking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "  OK: $dockerVersion" -ForegroundColor Green

    # Check if Docker daemon is running
    try {
        docker ps | Out-Null
        Write-Host "  OK: Docker daemon is running" -ForegroundColor Green
    } catch {
        Write-Host "  WARNING: Docker daemon not running" -ForegroundColor Yellow
        Write-Host "    Start Docker Desktop to continue" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ERROR: Docker not found" -ForegroundColor Red
    Write-Host "    Install from: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Red
    $allGood = $false
}

# Check Docker Compose
Write-Host "`nChecking Docker Compose..." -ForegroundColor Yellow
try {
    $composeVersion = docker-compose --version 2>&1
    Write-Host "  OK: $composeVersion" -ForegroundColor Green
} catch {
    try {
        $composeVersion = docker compose version 2>&1
        Write-Host "  OK: $composeVersion (plugin)" -ForegroundColor Green
    } catch {
        Write-Host "  WARNING: Docker Compose not found (optional)" -ForegroundColor Yellow
    }
}

# Check existing data
Write-Host "`nChecking Data..." -ForegroundColor Yellow

$dataChecks = @{
    "Pan-STARRS" = "bigG\bigG\hlsp_ps1cosmo_panstarrs_gpc1_all_model_v1_sys-full.txt"
    "Micro-fits" = "micro-bot-digest\micro-bot-digest"
    "HDGL Data" = "hdgl_harmonics_spiral10000_analog_v30\hdgl_spiral10000_v30.json"
    "LIGO Data" = "ligo_data"
}

foreach ($name in $dataChecks.Keys) {
    $path = $dataChecks[$name]
    if (Test-Path $path) {
        if ((Get-Item $path) -is [System.IO.DirectoryInfo]) {
            $count = (Get-ChildItem $path -Recurse -File).Count
            Write-Host "  OK: $name - $count files" -ForegroundColor Green
        } else {
            $size = (Get-Item $path).Length
            Write-Host "  OK: $name - $([math]::Round($size/1MB, 1)) MB" -ForegroundColor Green
        }
    } else {
        if ($name -eq "LIGO Data") {
            Write-Host "  MISSING: $name (run download_data.py)" -ForegroundColor Yellow
        } else {
            Write-Host "  ERROR: $name not found" -ForegroundColor Red
            $allGood = $false
        }
    }
}

# Check framework files
Write-Host "`nChecking Framework Files..." -ForegroundColor Yellow

$frameworkFiles = @(
    "grand_unified_theory.py",
    "gut_precision_engine.c",
    "gut_data_analysis.py",
    "gut_demo.py",
    "data_inventory.py",
    "download_data.py",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt"
)

$missingFiles = @()
foreach ($file in $frameworkFiles) {
    if (Test-Path $file) {
        Write-Host "  OK: $file" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: $file not found" -ForegroundColor Red
        $missingFiles += $file
        $allGood = $false
    }
}

# Check Python packages
Write-Host "`nChecking Python Packages..." -ForegroundColor Yellow

$packages = @("numpy", "scipy", "matplotlib", "pandas")
foreach ($pkg in $packages) {
    try {
        python -c "import $pkg; print('  OK: ' + '$pkg' + ' v' + $pkg.__version__)" 2>&1
    } catch {
        Write-Host "  WARNING: $pkg not installed" -ForegroundColor Yellow
        Write-Host "    Run: pip install $pkg" -ForegroundColor Yellow
    }
}

# Check GWOSC
Write-Host "`nChecking GWOSC (LIGO data tools)..." -ForegroundColor Yellow
try {
    python -c "import gwosc; print('  OK: gwosc v' + gwosc.__version__)" 2>&1
} catch {
    Write-Host "  MISSING: gwosc (for LIGO data download)" -ForegroundColor Yellow
    Write-Host "    Run: pip install gwosc" -ForegroundColor Yellow
}

# System resources
Write-Host "`nChecking System Resources..." -ForegroundColor Yellow

$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Write-Host "  RAM: $ram GB" -NoNewline
if ($ram -ge 8) {
    Write-Host " OK" -ForegroundColor Green
} elseif ($ram -ge 4) {
    Write-Host " WARNING (8+ GB recommended)" -ForegroundColor Yellow
} else {
    Write-Host " ERROR (minimum 4 GB)" -ForegroundColor Red
    $allGood = $false
}

$disk = Get-PSDrive C | Select-Object -ExpandProperty Free
$diskGB = [math]::Round($disk / 1GB, 1)
Write-Host "  Disk Space: $diskGB GB free" -NoNewline
if ($diskGB -ge 10) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " WARNING (10+ GB recommended)" -ForegroundColor Yellow
}

$cpus = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
Write-Host "  CPU Cores: $cpus" -NoNewline
if ($cpus -ge 4) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " WARNING (4+ cores recommended)" -ForegroundColor Yellow
}

# Summary
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

if ($allGood) {
    Write-Host "STATUS: Environment is READY!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. (Optional) Download LIGO data:" -ForegroundColor White
    Write-Host "     python download_data.py" -ForegroundColor Gray
    Write-Host "  2. Build Docker environment:" -ForegroundColor White
    Write-Host "     .\docker_build.ps1" -ForegroundColor Gray
    Write-Host "  3. Or run directly:" -ForegroundColor White
    Write-Host "     python grand_unified_theory.py" -ForegroundColor Gray
} else {
    Write-Host "STATUS: Environment has ISSUES" -ForegroundColor Red
    Write-Host "`nPlease fix the errors above before continuing." -ForegroundColor Yellow

    if ($missingFiles.Count -gt 0) {
        Write-Host "`nMissing files:" -ForegroundColor Yellow
        foreach ($file in $missingFiles) {
            Write-Host "  - $file" -ForegroundColor Red
        }
    }
}

Write-Host ""
