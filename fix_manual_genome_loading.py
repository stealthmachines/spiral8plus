#!/usr/bin/env python3
"""
Fix human_fasta files that manually load genome without respecting GENOME_LIMIT.
Replaces manual genome loading with proper load_genome() function.
"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8"

LOAD_GENOME_FUNC = '''
def load_genome(fasta_file, max_nucleotides=None, chromosome=None, start_position=0):
    """Load genome with GENOME_LIMIT support"""
    import os

    if max_nucleotides is None:
        max_nucleotides = int(os.environ.get('GENOME_LIMIT', '100000'))

    if chromosome is None:
        chromosome = os.environ.get('GENOME_CHROMOSOME', None)

    sequence = ""
    current_chr = None
    count = 0

    print(f"Loading genome (limit: {max_nucleotides:,})...")
    if chromosome:
        print(f"  Filtering: {chromosome}")

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                current_chr = line.strip()[1:].split()[0]
                if chromosome and current_chr != chromosome:
                    continue
                if chromosome:
                    print(f"  Loading from {current_chr}...")
            else:
                if chromosome and current_chr != chromosome:
                    continue

                bases = line.strip().upper()
                remaining = max_nucleotides - count

                if remaining <= 0:
                    break

                sequence += bases[:remaining]
                count += len(bases[:remaining])

                if count >= max_nucleotides:
                    break

    print(f"  Loaded: {count:,} nucleotides")
    return sequence

'''

def has_manual_genome_loading(content):
    """Check if file manually loads genome without load_genome()"""
    # Look for patterns like:
    # genome_seq.extend(list(line.strip().upper()))
    # genome_seq += line.strip()
    # seq.append(line.strip())
    patterns = [
        r'genome_seq\.extend\(.*line',
        r'genome_seq\s*\+=\s*line',
        r'seq\.append\(.*line',
        r'sequence\s*\+=\s*line.*strip'
    ]

    for pattern in patterns:
        if re.search(pattern, content):
            return True
    return False

def fix_manual_loading(filepath, filename):
    """Fix manual genome loading"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False, "Can't read"

    # Skip if already has load_genome function or doesn't manually load
    if 'def load_genome(' in content:
        return False, "Already has load_genome()"

    if not has_manual_genome_loading(content):
        return False, "No manual loading detected"

    # Find where to insert load_genome (after find_human_fasta)
    if 'def find_human_fasta():' in content:
        match = re.search(r'(def find_human_fasta\(\):.*?raise FileNotFoundError[^\n]+\n)', content, re.DOTALL)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + LOAD_GENOME_FUNC + content[insert_pos:]

            # Now replace manual loading code
            # Pattern 1: genome_seq = [] ... with open ... genome_seq.extend
            manual_pattern1 = re.compile(
                r'genome_seq\s*=\s*\[\]\s*\n.*?with\s+open\([^)]+\)\s+as\s+\w+:\s*\n.*?for\s+line\s+in\s+\w+:\s*\n.*?if\s+line\.startswith\(["\']>["\'].*?\n.*?genome_seq\.extend\(.*?\)\s*\ngenome_len\s*=\s*len\(genome_seq\)',
                re.DOTALL
            )

            if manual_pattern1.search(content):
                # Replace with load_genome call
                replacement = 'genome_seq = load_genome(fasta_path)\ngenome_len = len(genome_seq)\nprint(f"Genome length: {genome_len:,}")'
                content = manual_pattern1.sub(replacement, content)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Replaced manual loading with load_genome()"

    return False, "Couldn't apply fix"

def main():
    """Process all human_fasta files"""
    fixed_count = 0

    print("Fixing manual genome loading in human_fasta files...\n")

    for filename in sorted(os.listdir(workspace)):
        if not (filename.endswith('.py') and filename.startswith('human_fasta')):
            continue

        filepath = os.path.join(workspace, filename)
        modified, message = fix_manual_loading(filepath, filename)

        if modified:
            print(f"✓ {filename}: {message}")
            fixed_count += 1
        elif "Can't" not in message and "Already" not in message and "No manual" not in message:
            print(f"⚠ {filename}: {message}")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
