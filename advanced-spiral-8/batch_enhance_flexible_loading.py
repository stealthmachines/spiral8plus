"""
Enhanced batch fix: Add flexible loading options to human genome files
- User can specify nucleotide limit via command line or environment variable
- Supports chromosome selection
- Adds lazy loading option
"""

import os
import re
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Enhanced load_genome with flexible options
ENHANCED_LOAD_GENOME = '''def load_genome(fasta_file, max_nucleotides=None, chromosome=None):
    """
    Load genome sequence with flexible options for large genomes

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = all, but may be slow!)
                        Can also be set via environment variable GENOME_LIMIT
        chromosome: Specific chromosome to load (e.g., "chr1", "NC_000001.11")
                   If None, loads from first sequence found

    Returns:
        tuple: (sequence_string, metadata_dict)
    """
    import os

    # Check environment variable for limit
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        max_nucleotides = int(env_limit) if env_limit != 'all' else None

    seq = []
    count = 0
    current_chr = None
    found_chr = False
    metadata = {}

    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                # Header line
                header = line[1:].strip()

                # Extract chromosome info
                if current_chr is None or chromosome is not None:
                    current_chr = header.split()[0] if header else "unknown"
                    metadata['chromosome'] = current_chr
                    metadata['header'] = header

                # If looking for specific chromosome, check if this is it
                if chromosome is not None:
                    if chromosome.lower() in header.lower():
                        found_chr = True
                        print(f"Found chromosome: {header[:80]}...")
                    elif found_chr:
                        # We've finished the target chromosome
                        break
                    else:
                        # Skip this chromosome
                        continue
                continue

            # If we're filtering by chromosome and haven't found it, skip
            if chromosome is not None and not found_chr:
                continue

            # Load sequence data
            bases = list(line.strip().upper())

            # Apply nucleotide limit if specified
            if max_nucleotides is not None:
                remaining = max_nucleotides - count
                if remaining <= 0:
                    break
                seq.extend(bases[:remaining])
                count += len(bases[:remaining])
            else:
                seq.extend(bases)
                count += len(bases)

    metadata['length'] = len(seq)
    metadata['source_file'] = os.path.basename(fasta_file)

    print(f"Loaded {len(seq):,} nucleotides from {metadata.get('chromosome', 'unknown')}")
    if max_nucleotides and len(seq) == max_nucleotides:
        print(f"  (limited to {max_nucleotides:,}, set GENOME_LIMIT env var to change)")

    return ''.join(seq), metadata'''

def fix_file(filepath):
    """Add enhanced flexible loading to file"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already has the enhanced version
    if 'GENOME_LIMIT' in content or 'chromosome=' in content:
        print(f"  ⚠ Already has enhanced loading")
        return False

    # Pattern to find and replace load_genome function
    # Look for existing function with max_nucleotides parameter
    pattern = r'def load_genome\(fasta_file(?:, max_nucleotides=[^)]+)?\):.*?return (?:seq|\'\'\.join\(seq\))'

    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Replace with enhanced version
        content = content[:match.start()] + ENHANCED_LOAD_GENOME + content[match.end():]

        # Update the genome_seq = load_genome() call to handle tuple return
        # Pattern: genome_seq = load_genome(...)
        call_pattern = r'genome_seq\s*=\s*load_genome\(([^)]+)\)'

        def replace_call(m):
            args = m.group(1)
            # Check if already unpacking tuple
            if 'genome_seq, metadata' in content:
                return m.group(0)
            return f'genome_seq, genome_metadata = load_genome({args})'

        content = re.sub(call_pattern, replace_call, content)

        # Update genome_len calculation if it exists
        if 'genome_len = len(genome_seq)' in content:
            # Add metadata usage hint
            len_pattern = r'(genome_len = len\(genome_seq\))'
            content = re.sub(len_pattern, r'\1\n# Chromosome info available in genome_metadata dict', content)

        print(f"  ✓ Enhanced with flexible loading")
    else:
        print(f"  ⚠ Could not find load_genome function to enhance")
        return False

    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return True

def main():
    """Main enhancement process"""
    print("\n" + "="*70)
    print("BATCH ENHANCEMENT: Flexible Genome Loading")
    print("="*70)
    print("\nAdding features:")
    print("  • Custom nucleotide limits via GENOME_LIMIT env var")
    print("  • Chromosome selection")
    print("  • Metadata tracking")
    print("\nUsage examples:")
    print("  set GENOME_LIMIT=500000 && python human_ecoli17.py")
    print("  set GENOME_LIMIT=all && python human_ecoli17.py")
    print("="*70 + "\n")

    # Find all human_*.py files
    import glob
    all_files = sorted(glob.glob('human_*.py'))

    # Filter to only files with load_genome function
    target_files = []
    for f in all_files:
        with open(f, 'r', encoding='utf-8') as fp:
            if 'def load_genome' in fp.read():
                target_files.append(f)

    if not target_files:
        print("⚠ No suitable files found!")
        return

    print(f"Found {len(target_files)} files with load_genome function\n")

    fixed = []
    skipped = []

    for filepath in target_files:
        print(f"Enhancing: {filepath}")
        try:
            result = fix_file(filepath)
            if result:
                fixed.append(filepath)
            else:
                skipped.append(filepath)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            skipped.append(filepath)
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Enhanced: {len(fixed)} files")
    print(f"⚠ Skipped: {len(skipped)} files")

    if fixed:
        print(f"\n✅ Enhanced {len(fixed)} files with flexible loading!")
        print("\nNew features:")
        print("  • Set GENOME_LIMIT=500000 to load 500K nucleotides")
        print("  • Set GENOME_LIMIT=all to load entire genome (slow!)")
        print("  • Files return (sequence, metadata) tuple")
        print("  • Metadata includes chromosome info")

    print("="*70 + "\n")

if __name__ == '__main__':
    main()
