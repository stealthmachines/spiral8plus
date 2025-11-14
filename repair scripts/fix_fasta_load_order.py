#!/usr/bin/env python3
"""
Fix human_fasta files that call load_genome() before it's defined.
This causes them to load the entire genome instead of respecting GENOME_LIMIT.
"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8"

def fix_load_order(filepath, filename):
    """Move genome loading after load_genome() function definition"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False, "Can't read"

    # Check if it has the problematic pattern
    if 'genome_seq = load_genome(fasta_file) if "load_genome" in dir()' not in content:
        return False, "No problematic pattern"

    # Pattern: find the genome loading block
    load_pattern = re.compile(
        r'(# Load genome\n'
        r'fasta_file = find_human_fasta\(\)\n'
        r'genome_seq = load_genome\(fasta_file\) if "load_genome" in dir\(\) else .*?\n'
        r'genome_len = len\(genome_seq\)\n'
        r'print\(f"Genome loaded: {genome_len:,} nucleotides"\)\n)'
        r'\n+',
        re.DOTALL
    )

    # Pattern: find end of load_genome function
    func_end_pattern = re.compile(
        r'(    print\(f"  Loaded .*? nucleotides"\)\n'
        r'    return sequence\n)',
        re.DOTALL
    )

    load_match = load_pattern.search(content)
    func_match = func_end_pattern.search(content)

    if not load_match or not func_match:
        return False, "Can't find patterns"

    # Extract the loading code
    loading_code = load_match.group(1)

    # Remove it from original position
    content = load_pattern.sub('', content)

    # Insert after function definition
    insert_pos = func_match.end()
    new_loading = '\n# ---------- LOAD GENOME ----------\n' + loading_code.replace('# Load genome\n', '') + '\n'
    content = content[:insert_pos] + new_loading + content[insert_pos:]

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return True, "Fixed load order"

def main():
    """Process all human_fasta files"""
    fixed_count = 0
    files_to_fix = [
        'human_fasta2.py', 'human_fasta3.py', 'human_fasta4.py', 'human_fasta4b.py',
        'human_fasta4c.py', 'human_fasta4d.py', 'human_fasta4e.py', 'human_fasta4f.py',
        'human_fasta4g.py', 'human_fasta5.py', 'human_fasta6.py', 'human_fasta7.py',
        'human_fasta8.py', 'human_fasta14.py', 'human_fasta16.py', 'human_fasta17.py',
        'human_fasta17_covid.py', 'human_fasta18.py', 'human_fasta19.py'
    ]

    print("Fixing genome load order in human_fasta files...\n")

    for filename in sorted(files_to_fix):
        filepath = os.path.join(workspace, filename)
        if not os.path.exists(filepath):
            print(f"⊘ {filename}: File not found")
            continue

        modified, message = fix_load_order(filepath, filename)

        if modified:
            print(f"✓ {filename}: {message}")
            fixed_count += 1
        else:
            print(f"⚠ {filename}: {message}")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files - genome loading now respects GENOME_LIMIT")
    print("\nNext: Need to add canvas/view initialization to these files")

if __name__ == '__main__':
    main()
