@echo off
REM Build DNA Engine V2 (100% FASTA-Powered)

echo.
echo ========================================================================
echo    DNA Engine V2 - 100%% FASTA-Powered Build
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
echo Building dna_engine_v2.dll...
"%TCC_PATH%" -shared -o dna_engine_v2.dll dna_engine_v2.c

if %errorlevel% equ 0 (
    echo.
    echo ✓ Build successful: dna_engine_v2.dll
    echo.
    echo FASTA-DRIVEN FEATURES:
    echo   ✓ Camera motion (GC content, entropy)
    echo   ✓ Cell division (palindrome signatures)
    echo   ✓ Organelle spawn (local GC%%)
    echo   ✓ Color modulation (codon usage)
    echo   ✓ Physics strength (sequence entropy)
    echo.
    echo Run with:
    echo   python ecoli46_v2_100percent_fasta.py
    echo.
) else (
    echo.
    echo ✗ Build failed
)

pause
