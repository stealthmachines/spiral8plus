"""
Build script for DNA Engine V3 - uses Python's distutils for cross-platform compilation
"""

import sys
import os
from pathlib import Path

def build_with_distutils():
    """Build using Python's built-in compiler detection"""
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext
    import tempfile

    # Create temp setup
    module = Extension(
        'dna_engine_v3_temp',
        sources=['dna_engine_v3_pure_fasta.c'],
        extra_compile_args=['-O3'],
    )

    # Build in place
    original_dir = os.getcwd()
    try:
        setup(
            name='dna_engine_v3',
            ext_modules=[module],
            script_args=['build_ext', '--inplace'],
        )
        print("\n✓ Compilation successful!")

        # Find the compiled file
        for f in Path('.').glob('dna_engine_v3_temp.*'):
            if f.suffix in ['.pyd', '.so', '.dll']:
                # Rename to expected name
                target = 'dna_engine_v3_pure_fasta.dll' if sys.platform == 'win32' else 'dna_engine_v3_pure_fasta.so'
                if f.name != target:
                    import shutil
                    shutil.copy(str(f), target)
                print(f"✓ Created: {target}")
                return True
    except Exception as e:
        print(f"✗ Build failed: {e}")
        return False

    return False

def build_with_cffi():
    """Alternative: build using cffi"""
    try:
        from cffi import FFI
        ffibuilder = FFI()

        # Read the C source
        with open('dna_engine_v3_pure_fasta.c', 'r') as f:
            source = f.read()

        # Define the interface
        ffibuilder.cdef("""
            int init_engine(const char* fasta_path);
            int get_frame_data(int cell_id, int frame_num, void* strand1_out, void* strand2_out);
            int get_genome_length(void);
            int get_points_per_frame(void);
            int get_max_cells(void);
            double get_core_radius(void);
            void cleanup_engine(void);
        """)

        ffibuilder.set_source(
            "dna_engine_v3_cffi",
            source,
            extra_compile_args=['-O3'],
        )

        ffibuilder.compile(verbose=True)
        print("\n✓ CFFI compilation successful!")
        return True
    except ImportError:
        print("✗ CFFI not available")
        return False
    except Exception as e:
        print(f"✗ CFFI build failed: {e}")
        return False

if __name__ == '__main__':
    print("="*70)
    print("DNA Engine V3 - Build Script")
    print("="*70)

    # Try distutils first (most compatible)
    print("\nAttempting build with distutils...")
    if build_with_distutils():
        sys.exit(0)

    # Try CFFI as backup
    print("\nAttempting build with CFFI...")
    if build_with_cffi():
        sys.exit(0)

    print("\n✗ All build methods failed!")
    print("\nManual compilation commands:")
    print("  Windows (gcc): gcc -shared -o dna_engine_v3_pure_fasta.dll -O3 dna_engine_v3_pure_fasta.c -lm")
    print("  Windows (msvc): cl /LD /O2 dna_engine_v3_pure_fasta.c /Fe:dna_engine_v3_pure_fasta.dll")
    print("  Linux/Mac: gcc -shared -fPIC -o dna_engine_v3_pure_fasta.so -O3 dna_engine_v3_pure_fasta.c -lm")
    sys.exit(1)
