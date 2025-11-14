#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Build script for FASTA DNA Unified Framework V1
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════════════════"
echo "║         FASTA DNA Unified Framework V1 - Build Script                   ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Detect compiler
if command -v gcc &> /dev/null; then
    CC="gcc"
    echo "Compiler: GCC $(gcc --version | head -n1)"
elif command -v clang &> /dev/null; then
    CC="clang"
    echo "Compiler: Clang $(clang --version | head -n1)"
else
    echo "Error: No C compiler found (gcc or clang required)"
    exit 1
fi

# Source file
SRC="fasta_dna_unified_v1.c"
if [ ! -f "$SRC" ]; then
    echo "Error: Source file not found: $SRC"
    exit 1
fi

echo "Source: $SRC"
echo ""

# Build configurations
echo "Building release version..."
$CC -o fasta_dna_unified_v1 $SRC -lm -O3 -march=native -Wall
echo "✓ Release binary: fasta_dna_unified_v1"

echo ""
echo "Building debug version..."
$CC -o fasta_dna_unified_v1_debug $SRC -lm -g -Wall -Wextra -DDEBUG
echo "✓ Debug binary: fasta_dna_unified_v1_debug"

# Optional: Profile-guided optimization
if [ "$1" == "--pgo" ]; then
    echo ""
    echo "Building with profile-guided optimization..."

    # Step 1: Instrumented build
    $CC -o fasta_dna_unified_v1_pgo $SRC -lm -O3 -march=native -fprofile-generate

    # Step 2: Training run
    echo "Running training data..."
    ./fasta_dna_unified_v1_pgo ecoli_k12.fasta 1000 /dev/null

    # Step 3: Optimized build
    $CC -o fasta_dna_unified_v1 $SRC -lm -O3 -march=native -fprofile-use

    # Cleanup
    rm -f fasta_dna_unified_v1_pgo *.gcda

    echo "✓ PGO binary: fasta_dna_unified_v1"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "║                          BUILD COMPLETE                                  ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Usage:"
echo "  ./fasta_dna_unified_v1 [genome_file] [max_points] [output_file]"
echo ""
echo "Example:"
echo "  ./fasta_dna_unified_v1 ecoli_k12.fasta 8000 spiral_output.csv"
echo ""
