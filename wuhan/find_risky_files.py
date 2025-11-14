#!/usr/bin/env python3
"""Find files that use direct dictionary access for bases (potential KeyError on 'N')"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan"

# Pattern for direct dictionary access like: colors[base], base_colors[b], etc.
DIRECT_ACCESS_PATTERN = re.compile(r'\w+\[(?:base|b|nucleotide|nt)\]')

def check_file(filepath, filename):
    """Check if file uses direct dictionary access for bases"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for direct access
        has_direct = DIRECT_ACCESS_PATTERN.search(content) is not None
        has_get = '.get(' in content
        has_n = "'N':" in content or '"N":' in content

        return {
            'direct_access': has_direct,
            'uses_get': has_get,
            'has_n': has_n
        }
    except:
        return None

def main():
    risky_files = []
    safe_files = []

    # Check workspace root
    for filename in sorted(os.listdir(workspace)):
        if not (filename.endswith('.py') and (
            filename.startswith('eco') or
            filename.startswith('fasta') or
            filename.startswith('spiral') or
            filename.startswith('dna_echo') or
            filename.startswith('human_eco')
        )):
            continue

        filepath = os.path.join(workspace, filename)
        result = check_file(filepath, filename)

        if result and result['direct_access'] and not result['has_n']:
            risky_files.append(filename)
            print(f"⚠ RISKY: {filename} (direct access, no 'N')")
        elif result and result['uses_get']:
            safe_files.append(filename)

    # Check advanced-spiral-8
    spiral_dir = os.path.join(workspace, 'advanced-spiral-8')
    if os.path.exists(spiral_dir):
        for filename in sorted(os.listdir(spiral_dir)):
            if not (filename.endswith('.py') and filename.startswith('human_eco')):
                continue

            filepath = os.path.join(spiral_dir, filename)
            result = check_file(filepath, filename)

            if result and result['direct_access'] and not result['has_n']:
                risky_files.append(f"advanced-spiral-8/{filename}")
                print(f"⚠ RISKY: advanced-spiral-8/{filename} (direct access, no 'N')")
            elif result and result['uses_get']:
                safe_files.append(f"advanced-spiral-8/{filename}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Risky files (need 'N' fix): {len(risky_files)}")
    print(f"  Safe files (use .get()): {len(safe_files)}")

    if risky_files:
        print(f"\nRisky files list:")
        for f in risky_files[:20]:  # Show first 20
            print(f"  - {f}")
        if len(risky_files) > 20:
            print(f"  ... and {len(risky_files)-20} more")

if __name__ == '__main__':
    main()
