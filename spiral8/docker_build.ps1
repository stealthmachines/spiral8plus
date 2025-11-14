# Grand Unified Theory - Docker Build and Test Script (Windows)
# ==============================================================

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Blue
Write-Host "GUT Docker Build & Test" -ForegroundColor Blue
Write-Host "================================" -ForegroundColor Blue

# Check Docker availability
try {
    docker --version | Out-Null
    Write-Host "✓ Docker available" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker not found" -ForegroundColor Red
    Write-Host "  Please install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
}

# Check Docker Compose availability
try {
    docker-compose --version | Out-Null
    Write-Host "✓ Docker Compose available" -ForegroundColor Green
    $ComposeAvailable = $true
} catch {
    try {
        docker compose version | Out-Null
        Write-Host "✓ Docker Compose available (plugin)" -ForegroundColor Green
        $ComposeAvailable = $true
    } catch {
        Write-Host "⚠ Docker Compose not found (optional)" -ForegroundColor Yellow
        $ComposeAvailable = $false
    }
}

# Build image
Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Blue
docker build -t gut-testing:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Image built successfully" -ForegroundColor Green

# Create output directories
Write-Host ""
Write-Host "Creating output directories..." -ForegroundColor Blue
New-Item -ItemType Directory -Force -Path "output", "plots", "logs", "ligo_data" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Run tests
Write-Host ""
Write-Host "================================" -ForegroundColor Blue
Write-Host "Running Tests" -ForegroundColor Blue
Write-Host "================================" -ForegroundColor Blue

Write-Host ""
Write-Host "Test 1: Main Validation" -ForegroundColor Yellow
docker run --rm `
    -v "${PWD}/output:/gut/output" `
    gut-testing:latest `
    python grand_unified_theory.py

Write-Host ""
Write-Host "Test 2: Interactive Demo" -ForegroundColor Yellow
docker run --rm `
    -v "${PWD}/output:/gut/output" `
    -v "${PWD}/plots:/gut/plots" `
    gut-testing:latest `
    python gut_demo.py

Write-Host ""
Write-Host "Test 3: C Precision Engine" -ForegroundColor Yellow
docker run --rm `
    -v "${PWD}/output:/gut/output" `
    gut-testing:latest `
    gut_engine validate-all

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "All Tests Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

Write-Host ""
Write-Host "Output files:"
Write-Host "  - output\gut_report.json"
Write-Host "  - plots\*.png"
Write-Host "  - logs\*.log"

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Download LIGO data:"
Write-Host "     docker run --rm -it -v `"${PWD}/ligo_data:/gut/ligo_data`" gut-testing python download_data.py"
Write-Host ""
Write-Host "  2. Run full analysis:"
Write-Host "     docker run --rm -v `"${PWD}/output:/gut/output`" gut-testing python gut_data_analysis.py"
Write-Host ""
Write-Host "  3. Interactive shell:"
Write-Host "     docker run --rm -it gut-testing /bin/bash"
Write-Host ""
if ($ComposeAvailable) {
    Write-Host "  4. Run all services with Docker Compose:"
    Write-Host "     docker-compose up"
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
