"""
Add GENOME_LIMIT environment variable support to all human_eco and human_fasta scripts
"""

import os
import re
from pathlib import Path

# Enhanced load_genome function with full environment variable support
ENHANCED_LOAD_GENOME = '''def load_genome(fasta_file, max_nucleotides=None, chromosome=None, start_position=0):
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
                # New chromosome header
                header = line.strip()[1:].split()[0]
                current_chromosome = header
                position_in_chromosome = 0

                # If we're filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                # Reset skip flag for new chromosome
                if chromosome and current_chromosome == chromosome:
                    skip_until_start = start_position > 0
                    print(f"Loading from {current_chromosome}...")
            else:
                # If filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                bases = line.strip()

                # Handle start position skipping
                if skip_until_start:
                    if position_in_chromosome + len(bases) <= start_position:
                        position_in_chromosome += len(bases)
                        continue
                    else:
                        # Start is within this line
                        offset = start_position - position_in_chromosome
                        bases = bases[offset:]
                        position_in_chromosome = start_position
                        skip_until_start = False
                        print(f"  Started at position {start_position:,}")

                position_in_chromosome += len(bases)

                # Add nucleotides up to limit
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

def process_file(filepath):
    """Add GENOME_LIMIT support to a single file"""
    print(f"Processing {filepath.name}...")

    content = filepath.read_text(encoding='utf-8', errors='ignore')

    # Check if already has environment variable support
    if 'GENOME_LIMIT' in content:
        print(f"  ✓ Already has GENOME_LIMIT support, skipping")
        return False

    # Find existing load_genome function
    load_genome_pattern = r'def load_genome\([^)]*\):.*?(?=\ndef |\nclass |\n[A-Z]|\Z)'
    match = re.search(load_genome_pattern, content, re.DOTALL)

    if match:
        # Replace existing load_genome function
        old_function = match.group(0)
        new_content = content.replace(old_function, ENHANCED_LOAD_GENOME)

        # Ensure 'import os' is present
        if 'import os' not in new_content:
            # Add after first import
            new_content = re.sub(r'(import \w+)', r'\1\nimport os', new_content, count=1)

        filepath.write_text(new_content, encoding='utf-8')
        print(f"  ✓ Replaced load_genome function with enhanced version")
        return True
    else:
        print(f"  ⚠ No load_genome function found, skipping")
        return False

def main():
    workspace = Path(__file__).parent

    # Get all human_eco and human_fasta files
    eco_files = sorted(workspace.glob('human_eco*.py'))
    fasta_files = sorted(workspace.glob('human_fasta*.py'))

    all_files = eco_files + fasta_files

    print("="*70)
    print("ADD GENOME_LIMIT SUPPORT TO ALL ECO AND FASTA SCRIPTS")
    print("="*70)
    print()
    print(f"Found {len(eco_files)} eco scripts")
    print(f"Found {len(fasta_files)} fasta scripts")
    print(f"Total: {len(all_files)} scripts")
    print()

    updated = 0
    skipped = 0
    errors = 0

    for filepath in all_files:
        try:
            if process_file(filepath):
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors += 1

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Updated: {updated} files")
    print(f"Skipped: {skipped} files (already have support or no load_genome)")
    print(f"Errors: {errors} files")
    print()

    if updated > 0:
        print("✓ GENOME_LIMIT support added successfully!")
        print()
        print("All scripts now support:")
        print("  - GENOME_LIMIT: Limit nucleotides loaded")
        print("  - GENOME_CHROMOSOME: Load specific chromosome")
        print("  - GENOME_START: Start from specific position")

if __name__ == "__main__":
    main()
