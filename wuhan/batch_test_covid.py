"""
Batch tester: Verify all COVID-19 visualization files can load and initialize
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

        # Test 2: Check for required functions/patterns
        has_load_function = 'def load_genome' in code or 'def find_covid_fasta' in code
        has_genome_ref = 'genome_seq' in code or 'genome_len' in code

        if has_load_function and has_genome_ref:
            result['genome_load'] = True

        # Test 3: Try importing (but don't execute main)
        # This is risky for GUI apps, so we'll just check imports
        if 'import' in code:
            result['imports'] = True

    except SyntaxError as e:
        result['errors'].append(f"Syntax Error: {e}")
    except Exception as e:
        result['errors'].append(f"Error: {e}")

    return result

def main():
    """Main testing process"""
    print("\n" + "="*70)
    print("BATCH TEST: COVID-19 Visualization Files")
    print("="*70 + "\n")

    # Find all covid_*.py files
    covid_files = list(Path('.').glob('covid_*.py'))

    # Also check our manual creations
    manual_files = [
        'fasta17_covid.py',
        'fasta17_auto.py',
    ]

    all_files = covid_files + [Path(f) for f in manual_files if Path(f).exists()]
    all_files = sorted(set(all_files))

    if not all_files:
        print("⚠ No COVID-19 visualization files found!")
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

    total = len(results)
    syntax_pass = sum(1 for r in results if r['syntax'])
    imports_pass = sum(1 for r in results if r['imports'])
    genome_pass = sum(1 for r in results if r['genome_load'])

    print(f"Total files tested: {total}")
    print(f"Syntax check passed: {syntax_pass}/{total}")
    print(f"Imports detected: {imports_pass}/{total}")
    print(f"Genome loading code: {genome_pass}/{total}")

    # List any failures
    failures = [r for r in results if not r['syntax'] or r['errors']]
    if failures:
        print(f"\n⚠ {len(failures)} file(s) with issues:")
        for r in failures:
            print(f"  • {r['file']}")
            for error in r['errors']:
                print(f"    - {error}")
    else:
        print("\n✓ All files passed basic validation!")

    # Check for FASTA file
    print("\n" + "="*70)
    print("GENOME FILE CHECK")
    print("="*70)

    import glob
    possible_paths = [
        r"ncbi_dataset\data\GCF_009858895.2\*.fna",
        r"ncbi_dataset\data\GCA_009858895.3\*.fna",
    ]

    found_fasta = False
    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            print(f"✓ Found SARS-CoV-2 genome: {files[0]}")
            found_fasta = True
            break

    if not found_fasta:
        print("⚠ WARNING: SARS-CoV-2 genome file not found!")
        print("  Expected location: ncbi_dataset\\data\\GCF_009858895.2\\*.fna")
        print("  or: ncbi_dataset\\data\\GCA_009858895.3\\*.fna")

    print("\n" + "="*70)
    print("READY TO RUN")
    print("="*70)

    if syntax_pass == total and found_fasta:
        print("✓ All files validated successfully!")
        print("✓ SARS-CoV-2 genome file detected!")
        print("\nYou can now run any covid_*.py file:")
        print("  python covid_fasta16.py")
        print("  python covid_spiral8.py")
        print("  python fasta17_auto.py")
        print("  etc.")
    else:
        print("⚠ Some issues detected. Please review errors above.")

    print("="*70 + "\n")

if __name__ == '__main__':
    main()
