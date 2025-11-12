@echo off
REM Build DNA Engine Shared Library (Windows)

echo.
echo ========================================================================
echo    DNA Engine - C Shared Library Build
echo ========================================================================
echo.

if exist tcc\tcc.exe (
    echo Using TinyCC from .\tcc\
    set TCC_PATH=tcc\tcc.exe
) else if exist C:\tcc\tcc.exe (
    echo Using TinyCC from C:\tcc\
    set TCC_PATH=C:\tcc\tcc.exe
) else (
    echo TinyCC not found! Please run build_tcc.bat first.
    pause
    exit /b 1
)

echo.
echo Building dna_engine.dll (shared library)...
"%TCC_PATH%" -shared -o dna_engine.dll dna_engine.c

if %errorlevel% equ 0 (
    echo.
    echo ✓ Build successful: dna_engine.dll
    echo.
    echo Next steps:
    echo   1. python ecoli46_c_engine.py
    echo.
) else (
    echo.
    echo ✗ Build failed
)

pause
