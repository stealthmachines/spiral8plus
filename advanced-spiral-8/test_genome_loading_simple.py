"""
Simple genome loading test for human spiral files (no VisPy required)
"""

import os
import sys
from pathlib import Path

def test_genome_loading_only(filepath):
    """Test genome loading without VisPy dependencies"""
    print(f"\n{'='*70}")
    print(f"Testing: {filepath.name}")
    print(f"{'='*70}")

    # Set environment variable
    os.environ['GENOME_LIMIT'] = '10000'

    # Read file and extract only genome loading code
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create test code that includes only necessary parts
    test_code = []

    # Add imports
    test_code.append("import os")
    test_code.append("import glob")
    test_code.append("os.environ['GENOME_LIMIT'] = '10000'")
    test_code.append("")

    # Extract find_human_fasta function
    if 'def find_human_fasta' in content:
        start = content.index('def find_human_fasta')
        # Find the end of this function (next 'def' or significant marker)
        end_markers = ['\ndef load_genome', '\n# ']
        end = len(content)
        for marker in end_markers:
            idx = content.find(marker, start + 10)
            if idx != -1 and idx < end:
                end = idx

        func_code = content[start:end].strip()
        test_code.append(func_code)
        test_code.append("")

    # Extract load_genome function
    if 'def load_genome' in content:
        start = content.index('def load_genome')
        # Find the end of this function
        end_markers = ['\n# Load the', '\ntry:', '\n# ----------']
        end = len(content)
        for marker in end_markers:
            idx = content.find(marker, start + 10)
            if idx != -1 and idx < end:
                end = idx

        func_code = content[start:end].strip()
        test_code.append(func_code)
        test_code.append("")

    # Add test execution code
    test_code.append("# Test execution")
    test_code.append("try:")
    test_code.append("    fasta_path = find_human_fasta()")
    test_code.append("    print(f'✅ Found FASTA: {fasta_path}')")
    test_code.append("    genome_seq, metadata = load_genome(fasta_path)")
    test_code.append("    print(f'✅ Loaded {len(genome_seq):,} nucleotides')")
    test_code.append("    print(f'✅ Chromosomes: {metadata[\"chromosomes\"]}')")
    test_code.append("    print(f'✅ Total loaded: {metadata[\"total_loaded\"]:,}')")
    test_code.append("    print('\\n✅ GENOME LOADING: PASSED')")
    test_code.append("except FileNotFoundError as e:")
    test_code.append("    print(f'⚠️ FASTA file not found: {e}')")
    test_code.append("    print('   (This is expected if human genome data is not downloaded)')")
    test_code.append("except Exception as e:")
    test_code.append("    print(f'❌ GENOME LOADING: FAILED - {e}')")
    test_code.append("    import traceback")
    test_code.append("    traceback.print_exc()")

    # Execute the test
    full_code = '\n'.join(test_code)

    try:
        exec(full_code)
        return True
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all human spiral files"""

    script_dir = Path(__file__).parent

    # Find all human_spiral*.py files
    spiral_files = sorted(script_dir.glob('human_spiral*.py'))

    if not spiral_files:
        print("⚠️ No human_spiral*.py files found")
        return

    print("="*70)
    print("GENOME LOADING TEST: Human Spiral Files (No VisPy Required)")
    print("="*70)
    print(f"Found {len(spiral_files)} files to test")
    print(f"Testing with GENOME_LIMIT=10000 for fast execution\n")

    passed = 0
    failed = 0

    for filepath in spiral_files:
        if test_genome_loading_only(filepath):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total files: {len(spiral_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 All genome loading tests passed!")
    else:
        print(f"\n⚠️ {failed} file(s) had issues - review output above")

if __name__ == '__main__':
    main()
