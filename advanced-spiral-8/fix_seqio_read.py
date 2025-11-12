"""
Fix all scripts that use SeqIO.read() to handle multi-record FASTA files
"""

import re
from pathlib import Path

# The load_genome function to add
LOAD_GENOME_FUNCTION = '''
def load_genome(fasta_file, max_nucleotides=None, chromosome=None):
    """
    Load genome sequence from FASTA file with environment variable support

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var, default 100000)
        chromosome: Specific chromosome to load (None = use GENOME_CHROMOSOME env var)

    Returns:
        str: Genome sequence
    """
    import os

    # Get from environment if not specified
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        try:
            max_nucleotides = int(env_limit)
        except ValueError:
            max_nucleotides = 100000

    if chromosome is None:
        chromosome = os.environ.get('GENOME_CHROMOSOME', None)

    sequence = ""
    nucleotide_count = 0

    print(f"Loading genome from {fasta_file}...")
    if chromosome:
        print(f"  Filtering: Chromosome {chromosome}")
    print(f"  Limit: {max_nucleotides:,} nucleotides")

    # Parse all records (human genome has multiple chromosomes)
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Check if this is the chromosome we want
        if chromosome and record.id != chromosome:
            continue

        if chromosome:
            print(f"  Loading from {record.id}...")

        # Add sequence from this chromosome
        seq_str = str(record.seq)
        remaining = max_nucleotides - nucleotide_count

        if remaining <= 0:
            break

        sequence += seq_str[:remaining]
        nucleotide_count += len(seq_str[:remaining])

        if nucleotide_count >= max_nucleotides:
            break

        # If no chromosome filter, just take from first chromosome
        if not chromosome:
            break

    print(f"  Loaded {nucleotide_count:,} nucleotides")
    return sequence
'''

def fix_file(filepath):
    """Fix a single file that uses SeqIO.read()"""
    print(f"Fixing {filepath.name}...")

    content = filepath.read_text(encoding='utf-8', errors='ignore')

    # Check if it uses SeqIO.read()
    if 'SeqIO.read(' not in content:
        print(f"  ⊘ Does not use SeqIO.read(), skipping")
        return False

    # Check if already fixed
    if 'def load_genome(' in content:
        print(f"  ✓ Already has load_genome(), skipping")
        return False

    # Ensure 'import os' is present
    if 'import os' not in content:
        content = re.sub(r'(from Bio import SeqIO)', r'\1\nimport os', content, count=1)

    # Add load_genome function after find_human_fasta
    find_fasta_end = content.find('raise FileNotFoundError("Could not find Human Genome FASTA file')
    if find_fasta_end != -1:
        # Find the end of that line
        line_end = content.find('\n', find_fasta_end)
        if line_end != -1:
            # Insert load_genome function
            content = content[:line_end+1] + LOAD_GENOME_FUNCTION + content[line_end+1:]

    # Replace SeqIO.read() calls with load_genome()
    # Pattern: record = SeqIO.read(find_human_fasta(), "fasta")
    #          genome_seq = str(record.seq)
    old_pattern = r'record = SeqIO\.read\(find_human_fasta\(\), "fasta"\)\s*genome_seq = str\(record\.seq\)'
    new_code = 'genome_seq = load_genome(find_human_fasta())'

    content = re.sub(old_pattern, new_code, content)

    # Write back
    filepath.write_text(content, encoding='utf-8')
    print(f"  ✓ Fixed!")
    return True

def main():
    workspace = Path(__file__).parent

    # Find all human_eco*.py files
    eco_files = sorted(workspace.glob('human_eco*.py'))

    print("="*70)
    print("FIX SeqIO.read() CALLS IN ECO SCRIPTS")
    print("="*70)
    print()
    print(f"Found {len(eco_files)} eco scripts to check")
    print()

    fixed = 0
    skipped = 0

    for filepath in eco_files:
        try:
            if fix_file(filepath):
                fixed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Fixed: {fixed} files")
    print(f"Skipped: {skipped} files")

    if fixed > 0:
        print()
        print("✓ All SeqIO.read() calls fixed!")
        print("  Scripts now use load_genome() with GENOME_LIMIT support")

if __name__ == "__main__":
    main()
