"""
Comprehensive error check for all human_* files
Tests for common errors without running full visualization
"""

import os
import re
from pathlib import Path

def check_file_for_errors(filepath):
    """Check file for common errors"""

    errors = []
    warnings = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        lines = content.split('\n')

    # Check 1: .exists() on string (should use os.path.exists)
    if re.search(r'(\w+)\.exists\(\)', content):
        matches = re.findall(r'(\w+)\.exists\(\)', content)
        for match in matches:
            if match != 'Path' and match != 'path' and not match.endswith('_path'):
                errors.append(f"Using .exists() on potentially non-Path object: {match}")

    # Check 2: find_human_fasta exists and returns correctly
    if 'def find_human_fasta' not in content:
        warnings.append("No find_human_fasta() function found")
    else:
        if 'GCF_000001405.40' not in content:
            errors.append("Missing human genome accession (GCF_000001405.40)")
        if 'GCF_009858895' in content:
            errors.append("Still has COVID genome accession (GCF_009858895)")

    # Check 3: load_genome function
    if 'def load_genome' in content:
        # Check if it has GENOME_LIMIT support
        if 'GENOME_LIMIT' in content:
            warnings.append("Has GENOME_LIMIT support [OK]")
        else:
            warnings.append("Missing GENOME_LIMIT environment variable support")

    # Check 4: os module imported if using os.path
    if 'os.path' in content or 'os.environ' in content:
        if 'import os' not in content:
            errors.append("Uses os.path or os.environ but 'import os' not found")

    # Check 5: Proper encoding in file operations
    open_calls = re.findall(r'open\([^)]+\)', content)
    for call in open_calls:
        if 'encoding' not in call and '.fasta' in call or '.fna' in call:
            warnings.append(f"File open without explicit encoding: {call[:50]}...")
            break

    return errors, warnings

def main():
    """Check all human_* files"""

    script_dir = Path(__file__).parent

    # Find all human_*.py files (excluding test scripts)
    all_files = []
    all_files.extend(script_dir.glob('human_ecoli*.py'))
    all_files.extend(script_dir.glob('human_fasta*.py'))
    all_files.extend(script_dir.glob('human_spiral*.py'))

    all_files = sorted(set(all_files))

    print("="*70)
    print(f"ERROR CHECK: All Human Genome Files ({len(all_files)} files)")
    print("="*70)

    files_with_errors = []
    files_with_warnings = []
    clean_files = []

    for filepath in all_files:
        errors, warnings = check_file_for_errors(filepath)

        if errors:
            files_with_errors.append((filepath.name, errors, warnings))
            print(f"[ERR] {filepath.name}")
            for err in errors:
                print(f"   ERROR: {err}")
            for warn in warnings:
                print(f"   WARN: {warn}")
        elif warnings:
            files_with_warnings.append((filepath.name, warnings))
            print(f"[WARN] {filepath.name}")
            for warn in warnings:
                print(f"   {warn}")
        else:
            clean_files.append(filepath.name)
            print(f"[OK] {filepath.name}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total files: {len(all_files)}")
    print(f"Clean: {len(clean_files)}")
    print(f"With warnings: {len(files_with_warnings)}")
    print(f"With ERRORS: {len(files_with_errors)}")

    if files_with_errors:
        print(f"\n{'='*70}")
        print("FILES NEEDING FIXES:")
        print(f"{'='*70}")
        for filename, errors, warnings in files_with_errors:
            print(f"\n{filename}:")
            for err in errors:
                print(f"  ❌ {err}")

    if len(files_with_errors) == 0:
        print(f"\nSUCCESS: No critical errors found!")
        return True
    else:
        print(f"\nWARNING: {len(files_with_errors)} file(s) need attention")
        return False

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
