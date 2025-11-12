@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM Build script for FASTA DNA Unified Framework V1 (Windows)
REM ═══════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo ═══════════════════════════════════════════════════════════════════════════
echo ║         FASTA DNA Unified Framework V1 - Build Script (Windows)         ║
echo ═══════════════════════════════════════════════════════════════════════════
echo.

set SRC=fasta_dna_unified_v1.c
if not exist "%SRC%" (
    echo Error: Source file not found: %SRC%
    exit /b 1
)

echo Source: %SRC%
echo.

REM Try MSVC first
where cl >nul 2>&1
if %errorlevel% equ 0 (
    echo Compiler: Microsoft Visual C++
    echo.

    echo Building release version with MSVC...
    cl /O2 /fp:fast /W3 %SRC% /Fe:fasta_dna_unified_v1.exe
    if %errorlevel% equ 0 (
        echo ✓ Release binary: fasta_dna_unified_v1.exe
    ) else (
        echo ✗ Build failed
        exit /b 1
    )

    echo.
    echo Building debug version with MSVC...
    cl /Zi /Od /W3 /DDEBUG %SRC% /Fe:fasta_dna_unified_v1_debug.exe
    if %errorlevel% equ 0 (
        echo ✓ Debug binary: fasta_dna_unified_v1_debug.exe
    )

    goto :done
)

REM Try GCC (MinGW)
where gcc >nul 2>&1
if %errorlevel% equ 0 (
    echo Compiler: GCC (MinGW)
    echo.

    echo Building release version with GCC...
    gcc -o fasta_dna_unified_v1.exe %SRC% -lm -O3 -march=native -Wall
    if %errorlevel% equ 0 (
        echo ✓ Release binary: fasta_dna_unified_v1.exe
    ) else (
        echo ✗ Build failed
        exit /b 1
    )

    echo.
    echo Building debug version with GCC...
    gcc -o fasta_dna_unified_v1_debug.exe %SRC% -lm -g -Wall -Wextra -DDEBUG
    if %errorlevel% equ 0 (
        echo ✓ Debug binary: fasta_dna_unified_v1_debug.exe
    )

    goto :done
)

REM No compiler found
echo Error: No C compiler found
echo.
echo Please install one of the following:
echo   - Microsoft Visual C++ (MSVC) - Part of Visual Studio
echo   - GCC (MinGW-w64) - https://www.mingw-w64.org/
echo.
exit /b 1

:done
echo.
echo ═══════════════════════════════════════════════════════════════════════════
echo ║                          BUILD COMPLETE                                  ║
echo ═══════════════════════════════════════════════════════════════════════════
echo.
echo Usage:
echo   fasta_dna_unified_v1.exe [genome_file] [max_points] [output_file]
echo.
echo Example:
echo   fasta_dna_unified_v1.exe ecoli_k12.fasta 8000 spiral_output.csv
echo.

endlocal
