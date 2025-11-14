# Test Tuned Parameters in Docker
# ================================
# Builds and runs tuned echo parameter test in Docker environment

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DOCKER: Test Tuned Echo Parameters" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Docker is running
$dockerRunning = docker info 2>&1 | Select-String "Server Version"
if (-not $dockerRunning) {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host "   Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Docker is running" -ForegroundColor Green

# Build the image (if not already built)
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
$buildOutput = docker build -t gut-testing:latest . 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker image built successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    Write-Host $buildOutput -ForegroundColor Red
    exit 1
}

# Create output directories if they don't exist
$dirs = @("output", "plots", "logs", "ligo_data")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "✓ Created directory: $dir" -ForegroundColor Green
    }
}

# Run the tuned parameter test in Docker
Write-Host "`nRunning tuned parameter test in Docker..." -ForegroundColor Yellow
Write-Host "(This will download real LIGO data - may take a few minutes)`n" -ForegroundColor Cyan

docker run --rm `
    -v "${PWD}/output:/gut/output" `
    -v "${PWD}/plots:/gut/plots" `
    -v "${PWD}/logs:/gut/logs" `
    -v "${PWD}/ligo_data:/gut/ligo_data" `
    -v "${PWD}/tuned_echo_parameters.json:/gut/tuned_echo_parameters.json:ro" `
    -v "${PWD}/test_tuned_ligo.py:/gut/test_tuned_ligo.py:ro" `
    -e PYTHONUNBUFFERED=1 `
    gut-testing:latest `
    python test_tuned_ligo.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Test completed successfully!" -ForegroundColor Green
    Write-Host "`nResults saved to:" -ForegroundColor Cyan
    Write-Host "  - tuned_ligo_test_results.json (in output/)" -ForegroundColor White
    Write-Host "  - tuned_echo_comparison.png (in plots/)" -ForegroundColor White
} else {
    Write-Host "`n❌ Test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Docker test complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
