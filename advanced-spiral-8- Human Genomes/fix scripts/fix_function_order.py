"""
Fix function definition order: Move find_covid_fasta() before it's called
"""

import re
import os

FILES_TO_FIX = [
    'covid_ecoli16.py',
    'covid_ecoli17.py',
    'covid_ecoli19.py',
    'covid_ecoli20.py',
    'covid_ecoli21.py',
]

def fix_file(filepath):
    """Move find_covid_fasta definition before its first use"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the function definition
    func_pattern = r'# ---------- AUTO-DETECT AND LOAD GENOME ----------\ndef find_covid_fasta\(\):.*?raise FileNotFoundError\([^)]+\)\n'
    func_match = re.search(func_pattern, content, re.DOTALL)

    if not func_match:
        print(f"  ⚠ Could not find function in {filepath}")
        return False

    func_code = func_match.group(0)

    # Remove the function from its current location
    content = content.replace(func_code, '')

    # Find the imports section and add function after it
    import_pattern = r'((?:^(?:from|import)\s+.*\n)+)'
    match = re.search(import_pattern, content, re.MULTILINE)

    if match:
        last_import_end = match.end()
        content = content[:last_import_end] + '\n' + func_code + '\n' + content[last_import_end:]
    else:
        # No imports found, add at beginning
        content = func_code + '\n' + content

    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return True

def main():
    print("\n" + "="*70)
    print("FIXING FUNCTION DEFINITION ORDER")
    print("="*70 + "\n")

    fixed = []
    failed = []

    for filename in FILES_TO_FIX:
        if not os.path.exists(filename):
            print(f"⚠ Skipping {filename} (not found)\n")
            failed.append(filename)
            continue

        print(f"Fixing: {filename}")
        if fix_file(filename):
            fixed.append(filename)
            print(f"  ✓ Success\n")
        else:
            failed.append(filename)
            print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Fixed: {len(fixed)} files")
    print(f"✗ Failed: {len(failed)} files")

    if fixed:
        print("\nFixed files:")
        for f in fixed:
            print(f"  • {f}")

    print("="*70 + "\n")

if __name__ == '__main__':
    main()
