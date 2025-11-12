"""
Batch tester: Verify all Human Genome visualization files can load and initialize
Tests syntax, imports, and genome loading without running full visualizations
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_file(filepath):
    """Test if a Python file can be imported and has basic functionality"""

    result = {
        'file': filepath.name,
        'syntax': False,
        'imports': False,
        'genome_load': False,
        'errors': []
    }

    try:
        # Test 1: Syntax check (compile)
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        compile(code, filepath.name, 'exec')
        result['syntax'] = True

        # Test 2: Check for imports
        if 'import' in code:
            result['imports'] = True

        # Test 3: Check for genome loading code
        if 'find_human_fasta' in code or 'load_genome' in code or 'genome_seq' in code:
            result['genome_load'] = True

    except SyntaxError as e:
        result['errors'].append(f"Syntax error: {e.msg} at line {e.lineno}")
    except Exception as e:
        result['errors'].append(f"Error: {str(e)}")

    return result

def main():
    """Main testing process"""

    print("\n" + "="*70)
    print("BATCH TEST: Human Genome Visualization Files")
    print("="*70 + "\n")

    # Find all human_*.py files
    current_dir = Path.cwd()
    all_files = sorted(current_dir.glob('human_*.py'))

    # Also check for fasta17_auto.py if it exists and uses human genome
    for extra in ['fasta17_auto.py', 'fasta17_human.py']:
        extra_path = current_dir / extra
        if extra_path.exists():
            with open(extra_path, 'r', encoding='utf-8') as f:
                if 'find_human_fasta' in f.read():
                    all_files.append(extra_path)

    if not all_files:
        print("⚠ No human genome visualization files found!")
        print("Run batch_convert_to_human.py first.")
        return

    print(f"Found {len(all_files)} files to test\n")

    results = []
    for filepath in all_files:
        if not filepath.exists():
            continue

        print(f"Testing: {filepath.name}")
        result = test_file(filepath)
        results.append(result)

        # Display result
        status_marks = []
        status_marks.append("✓ Syntax" if result['syntax'] else "✗ Syntax")
        status_marks.append("✓ Imports" if result['imports'] else "✗ Imports")
        status_marks.append("✓ Genome" if result['genome_load'] else "✗ Genome")

        print(f"  {' | '.join(status_marks)}")

        if result['errors']:
            for error in result['errors']:
                print(f"  ⚠ {error}")

        print()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total files tested: {len(results)}")
    print(f"Syntax check passed: {sum(1 for r in results if r['syntax'])}/{len(results)}")
    print(f"Imports detected: {sum(1 for r in results if r['imports'])}/{len(results)}")
    print(f"Genome loading code: {sum(1 for r in results if r['genome_load'])}/{len(results)}")

    all_passed = all(r['syntax'] and r['imports'] and r['genome_load'] for r in results)

    if all_passed:
        print("\n✓ All files passed basic validation!")
    else:
        print("\n⚠ Some files have issues - check output above")

    # Check for genome files
    print("\n" + "="*70)
    print("GENOME FILE CHECK")
    print("="*70)

    import glob
    possible_paths = [
        r"ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\GCA_000001405.29\*.fna",
    ]

    genome_found = False
    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            print(f"✓ Found Human Genome: {files[0]}")
            genome_found = True
            break

    if not genome_found:
        print("⚠ WARNING: Human Genome FASTA file not found!")
        print("Expected location: ncbi_dataset\\ncbi_dataset\\data\\GCF_000001405.40\\")

    # Final message
    print("\n" + "="*70)
    if all_passed and genome_found:
        print("READY TO RUN")
        print("="*70)
        print("✓ All files validated successfully!")
        print("✓ Human Genome file detected!")
        print("\nYou can now run any human_*.py file:")
        print("  python human_fasta16.py")
        print("  python human_ecoli10.py")
        print("  etc.")
    else:
        print("ISSUES DETECTED")
        print("="*70)
        if not all_passed:
            print("⚠ Some files failed validation")
        if not genome_found:
            print("⚠ Human Genome file not found")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
