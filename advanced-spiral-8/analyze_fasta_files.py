"""
Check all fasta*.py files to see which ones load genome data and need human versions
"""

import os
import re
from pathlib import Path

def check_file_for_genome_loading(filepath):
    """Check if a file loads genome/FASTA data"""

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Indicators that a file loads genome data
    genome_indicators = [
        r'\.fasta',
        r'\.fna',
        r'ecoli',
        r'genome',
        r'SeqIO',
        r'load.*fasta',
        r'FASTA',
        r'DNA.*sequence',
    ]

    for pattern in genome_indicators:
        if re.search(pattern, content, re.IGNORECASE):
            return True, pattern

    return False, None

def main():
    """Check all fasta files"""

    script_dir = Path(__file__).parent

    # Get all fasta files
    all_fasta_files = sorted(script_dir.glob('fasta*.py'))

    # Get existing human_fasta files
    existing_human = set(f.name.replace('human_', '') for f in script_dir.glob('human_fasta*.py'))

    print("="*70)
    print("FASTA FILES ANALYSIS")
    print("="*70)
    print(f"\nTotal fasta*.py files: {len(all_fasta_files)}")
    print(f"Existing human_fasta*.py files: {len(existing_human)}\n")

    needs_conversion = []
    no_genome = []
    already_has_human = []

    for filepath in all_fasta_files:
        has_genome, pattern = check_file_for_genome_loading(filepath)

        if filepath.name in existing_human:
            already_has_human.append(filepath.name)
        elif has_genome:
            needs_conversion.append((filepath.name, pattern))
        else:
            no_genome.append(filepath.name)

    # Report
    if needs_conversion:
        print(f"{'='*70}")
        print(f"FILES THAT NEED HUMAN VERSIONS ({len(needs_conversion)}):")
        print(f"{'='*70}")
        for filename, pattern in needs_conversion:
            print(f"  • {filename:30} (found: {pattern})")

    if already_has_human:
        print(f"\n{'='*70}")
        print(f"FILES THAT ALREADY HAVE HUMAN VERSIONS ({len(already_has_human)}):")
        print(f"{'='*70}")
        for filename in already_has_human:
            print(f"  ✅ {filename}")

    if no_genome:
        print(f"\n{'='*70}")
        print(f"FILES WITHOUT GENOME LOADING ({len(no_genome)}):")
        print(f"{'='*70}")
        for filename in no_genome:
            print(f"  ⚪ {filename}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Need conversion: {len(needs_conversion)}")
    print(f"Already have human version: {len(already_has_human)}")
    print(f"No genome loading: {len(no_genome)}")
    print(f"Total: {len(all_fasta_files)}")

    # Create list for batch converter
    if needs_conversion:
        print(f"\n{'='*70}")
        print("FILES TO ADD TO BATCH CONVERTER:")
        print(f"{'='*70}")
        for filename, _ in needs_conversion:
            print(f"    '{filename}',")

if __name__ == '__main__':
    main()
