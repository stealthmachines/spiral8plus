@echo off
REM TinyCC Build Script - No installation required
REM Download TCC from: https://bellard.org/tcc/

echo.
echo ========================================================================
echo    FASTA DNA Unified Framework V1 - TinyCC Quick Build
echo ========================================================================
echo.

if exist tcc\tcc.exe (
    echo Using TinyCC from .\tcc\
    set TCC_PATH=tcc\tcc.exe
) else if exist C:\tcc\tcc.exe (
    echo Using TinyCC from C:\tcc\
    set TCC_PATH=C:\tcc\tcc.exe
) else (
    echo TinyCC not found!
    echo.
    echo Please download TinyCC:
    echo   1. Visit: https://bellard.org/tcc/
    echo   2. Download "tcc-0.9.27-win64-bin.zip"
    echo   3. Extract to .\tcc\ or C:\tcc\
    echo.
    echo Or use the auto-download option below:
    echo.
    choice /C YN /M "Auto-download TinyCC now (7MB)"
    if errorlevel 2 goto :end
    if errorlevel 1 goto :download
)

:compile
echo.
echo Building fasta_dna_unified_v1.exe with TinyCC...
"%TCC_PATH%" -o fasta_dna_unified_v1.exe fasta_dna_unified_v1.c

if %errorlevel% equ 0 (
    echo.
    echo ✓ Build successful!
    echo.
    echo Run with:
    echo   fasta_dna_unified_v1.exe ecoli_k12.fasta 8000 spiral_output.csv
) else (
    echo.
    echo ✗ Build failed
)
goto :end

:download
echo.
echo Downloading TinyCC...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27-win64-bin.zip' -OutFile 'tcc.zip' }"

if %errorlevel% neq 0 (
    echo Download failed. Please download manually from https://bellard.org/tcc/
    goto :end
)

echo Extracting TinyCC...
powershell -Command "& { Expand-Archive -Path 'tcc.zip' -DestinationPath '.' -Force }"

if exist tcc (
    echo TinyCC installed successfully!
    set TCC_PATH=tcc\tcc.exe
    del tcc.zip
    goto :compile
) else (
    echo Extraction failed. Please extract manually.
    goto :end
)

:end
echo.
pause
