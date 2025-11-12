#!/usr/bin/env python3
"""
Fix human_fasta*.py files that are missing find_human_fasta() or genome loading.
"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8"

FIND_HUMAN_FASTA = '''def find_human_fasta():
    """Automatically find the Human Genome FASTA file in the ncbi_dataset"""
    import glob
    possible_paths = [
        r"ncbi_dataset\\ncbi_dataset\\data\\GCF_000001405.40\\*.fna",
        r"ncbi_dataset\\ncbi_dataset\\data\\GCA_000001405.29\\*.fna",
        r"ncbi_dataset\\ncbi_dataset\\data\\*\\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find Human Genome FASTA file in ncbi_dataset directory")

'''

def fix_fasta_file(filepath, filename):
    """Fix a human_fasta file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False, "Can't read"

    modified = False
    fixes = []

    # Check if file calls find_human_fasta() but doesn't define it
    if 'find_human_fasta()' in content and 'def find_human_fasta():' not in content:
        # Insert at beginning after imports
        lines = content.split('\n')
        insert_idx = 0

        # Find last import line
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_idx = i + 1

        # Insert function after imports
        lines.insert(insert_idx, '\n' + FIND_HUMAN_FASTA)
        content = '\n'.join(lines)
        modified = True
        fixes.append("Added find_human_fasta()")

    # Check if file uses genome_seq but doesn't load it
    if 'genome_seq' in content and 'genome_seq =' not in content:
        # Find where to insert (after imports or after find_human_fasta)
        if 'def find_human_fasta():' in content:
            # Insert after find_human_fasta function
            match = re.search(r'(def find_human_fasta\(\):.*?raise FileNotFoundError[^\n]+\n)', content, re.DOTALL)
            if match:
                insert_pos = match.end()
                genome_load = '\n# Load genome\nfasta_file = find_human_fasta()\ngenome_seq = load_genome(fasta_file) if "load_genome" in dir() else open(fasta_file).read().replace("\\n", "").upper()\ngenome_len = len(genome_seq)\nprint(f"Genome loaded: {genome_len:,} nucleotides")\n'
                content = content[:insert_pos] + genome_load + content[insert_pos:]
                modified = True
                fixes.append("Added genome loading")

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, ", ".join(fixes)

    return False, "No changes needed"

def main():
    """Process all human_fasta files"""
    fixed_count = 0

    print("Fixing human_fasta*.py files...\n")

    for filename in sorted(os.listdir(workspace)):
        if not (filename.endswith('.py') and filename.startswith('human_fasta')):
            continue

        filepath = os.path.join(workspace, filename)
        modified, message = fix_fasta_file(filepath, filename)

        if modified:
            print(f"✓ {filename}: {message}")
            fixed_count += 1

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
