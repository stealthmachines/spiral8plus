#!/usr/bin/env python3
"""
Comprehensive fix for 'N' nucleotide support across all script types.
Handles multiple color dictionary patterns and architectures.
"""
import os
import re

# Get workspace root
workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan"

# Pattern 1: base_colors dictionary (most scripts)
PATTERN1_SEARCH = re.compile(
    r"(base_colors\s*=\s*\{[^}]*'T':\s*\([^)]+\),?)",
    re.MULTILINE | re.DOTALL
)

PATTERN1_N_ENTRY = "    'N': (0.5, 0.5, 0.5, 1),  # Gray (unknown)"

# Pattern 2: base_map dictionary (geometry-based scripts)
PATTERN2_SEARCH = re.compile(
    r"(base_map\s*=\s*\{[^}]*'C':\s*\d+,)",
    re.MULTILINE | re.DOTALL
)

PATTERN2_N_ENTRY = "    'N': 1,  # Point (unknown)"

def fix_file(filepath):
    """Fix a single file, return True if modified"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already has 'N'
    if "'N':" in content or '"N":' in content:
        return False

    modified = False

    # Try Pattern 1: base_colors
    if 'base_colors' in content:
        match = PATTERN1_SEARCH.search(content)
        if match:
            # Insert N entry after T entry
            insert_pos = match.end(1)
            content = content[:insert_pos] + "\n" + PATTERN1_N_ENTRY + content[insert_pos:]
            modified = True
            print(f"  → Added 'N' to base_colors")

    # Try Pattern 2: base_map (geometry-based)
    elif 'base_map' in content:
        match = PATTERN2_SEARCH.search(content)
        if match:
            # Insert N entry after C entry
            insert_pos = match.end(1)
            content = content[:insert_pos] + "\n" + PATTERN2_N_ENTRY + content[insert_pos:]
            modified = True
            print(f"  → Added 'N' to base_map")

    # Save if modified
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    """Process all Python visualization scripts"""
    fixed_count = 0
    skipped_count = 0
    no_dict_count = 0

    # Process files in workspace root
    for filename in sorted(os.listdir(workspace)):
        if not (filename.endswith('.py') and (
            filename.startswith('eco') or
            filename.startswith('fasta') or
            filename.startswith('spiral') or
            filename.startswith('dna_echo')
        )):
            continue

        filepath = os.path.join(workspace, filename)
        print(f"\n{filename}:")

        result = fix_file(filepath)
        if result:
            fixed_count += 1
        elif "'N':" in open(filepath, 'r', encoding='utf-8').read():
            print(f"  ✓ Already has 'N'")
            skipped_count += 1
        else:
            print(f"  ⚠ No base_colors or base_map found")
            no_dict_count += 1

    # Process files in advanced-spiral-8
    spiral_dir = os.path.join(workspace, 'advanced-spiral-8')
    if os.path.exists(spiral_dir):
        for filename in sorted(os.listdir(spiral_dir)):
            if not (filename.endswith('.py') and filename.startswith('human_eco')):
                continue

            filepath = os.path.join(spiral_dir, filename)
            print(f"\n{filename}:")

            result = fix_file(filepath)
            if result:
                fixed_count += 1
            elif "'N':" in open(filepath, 'r', encoding='utf-8').read():
                print(f"  ✓ Already has 'N'")
                skipped_count += 1
            else:
                print(f"  ⚠ No base_colors or base_map found")
                no_dict_count += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Fixed: {fixed_count} files")
    print(f"  Already had 'N': {skipped_count} files")
    print(f"  No color dict: {no_dict_count} files")
    print(f"  Total processed: {fixed_count + skipped_count + no_dict_count}")
    print(f"\n✓ 'N' nucleotide support added!")

if __name__ == '__main__':
    main()
