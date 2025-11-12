#!/usr/bin/env python3
"""
Fix all remaining issues in human_eco scripts:
1. Missing find_human_fasta() and load_genome() functions
2. Missing canvas/view initialization
3. Function call before definition
"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8"

# Standard function definitions to add
FIND_HUMAN_FASTA = '''# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_human_fasta():
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


def load_genome(fasta_file, max_nucleotides=None, chromosome=None, start_position=0):
    """
    Load genome sequence from FASTA file with environment variable support

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var, default 100000)
        chromosome: Specific chromosome to load (None = use GENOME_CHROMOSOME env var)
        start_position: Starting position in sequence (default 0, or GENOME_START env var)

    Returns:
        str: Genome sequence
    """
    import os
    from Bio import SeqIO

    # Get from environment if not specified
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        try:
            max_nucleotides = int(env_limit)
        except ValueError:
            max_nucleotides = 100000

    if chromosome is None:
        chromosome = os.environ.get('GENOME_CHROMOSOME', None)

    if start_position == 0:
        env_start = os.environ.get('GENOME_START', '0')
        try:
            start_position = int(env_start)
        except ValueError:
            start_position = 0

    sequence = ""
    current_chromosome = None
    nucleotide_count = 0
    position_in_chromosome = 0
    skip_until_start = start_position > 0

    print(f"Loading genome from {fasta_file}...")
    if chromosome:
        print(f"  Filtering: Chromosome {chromosome}")
    if start_position > 0:
        print(f"  Starting at position: {start_position:,}")
    print(f"  Limit: {max_nucleotides:,} nucleotides")

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip()[1:].split()[0]
                current_chromosome = header
                position_in_chromosome = 0

                if chromosome and current_chromosome != chromosome:
                    continue

                if chromosome and current_chromosome == chromosome:
                    skip_until_start = start_position > 0
                    print(f"Loading from {current_chromosome}...")
            else:
                if chromosome and current_chromosome != chromosome:
                    continue

                bases = line.strip()

                if skip_until_start:
                    if position_in_chromosome + len(bases) <= start_position:
                        position_in_chromosome += len(bases)
                        continue
                    else:
                        offset = start_position - position_in_chromosome
                        bases = bases[offset:]
                        position_in_chromosome = start_position
                        skip_until_start = False
                        print(f"  Started at position {start_position:,}")

                position_in_chromosome += len(bases)
                remaining = max_nucleotides - nucleotide_count

                if remaining <= 0:
                    break

                sequence += bases[:remaining]
                nucleotide_count += len(bases[:remaining])

                if nucleotide_count >= max_nucleotides:
                    break

            if nucleotide_count >= max_nucleotides:
                break

    print(f"  Loaded {nucleotide_count:,} nucleotides")
    return sequence

'''

def fix_file(filepath, filename):
    """Fix issues in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False, "Can't read file"

    modified = False
    issues_fixed = []

    # Issue 1: Check if file calls find_human_fasta() before defining it
    if 'find_human_fasta()' in content and 'def find_human_fasta():' not in content:
        # Find where to insert (after imports, before first function call)
        import_end = 0
        for match in re.finditer(r'^import |^from ', content, re.MULTILINE):
            import_end = max(import_end, match.end())

        # Find first line after imports
        lines = content.split('\n')
        insert_line = 0
        for i, line in enumerate(lines):
            if i * len(line) > import_end:
                insert_line = i
                break

        # Insert functions
        lines.insert(insert_line, '\n' + FIND_HUMAN_FASTA)
        content = '\n'.join(lines)
        modified = True
        issues_fixed.append("Added find_human_fasta() and load_genome()")

    # Issue 2: Check if load_genome is defined but find_human_fasta is not
    if 'def load_genome(' in content and 'def find_human_fasta():' not in content:
        # Insert before load_genome
        insert_pos = content.find('def load_genome(')
        find_func = FIND_HUMAN_FASTA.split('def load_genome(')[0]
        content = content[:insert_pos] + find_func + content[insert_pos:]
        modified = True
        issues_fixed.append("Added find_human_fasta()")

    # Issue 3: Missing canvas/view initialization
    if 'canvas.show()' in content and 'canvas = scene.SceneCanvas' not in content:
        # Need to add canvas initialization after genome loading
        if 'genome_seq =' in content or 'genome_len =' in content:
            # Find where genome is loaded
            match = re.search(r'(genome_len\s*=\s*len\([^)]+\))', content)
            if match:
                insert_pos = match.end()
                canvas_init = '''

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
'''
                content = content[:insert_pos] + canvas_init + content[insert_pos:]
                modified = True
                issues_fixed.append("Added canvas/view initialization")

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, ", ".join(issues_fixed)

    return False, "No issues found"

def main():
    """Process all human_eco files"""
    fixed_count = 0

    for filename in sorted(os.listdir(workspace)):
        if not (filename.endswith('.py') and filename.startswith('human_eco')):
            continue

        filepath = os.path.join(workspace, filename)
        modified, message = fix_file(filepath, filename)

        if modified:
            print(f"✓ {filename}: {message}")
            fixed_count += 1
        elif "Can't read" in message:
            print(f"⚠ {filename}: {message}")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
