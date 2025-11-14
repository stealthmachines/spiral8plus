"""
Batch convert COVID spiral visualization files to human genome versions
Converts: covid_spiral8.py, covid_spiral9.py -> human_spiral8.py, human_spiral9.py
"""

import os
import re
from pathlib import Path

def convert_to_human(covid_file, human_file):
    """Convert a single COVID spiral file to human genome version"""

    print(f"\n{'='*60}")
    print(f"Converting: {covid_file.name}")
    print(f"       To: {human_file.name}")
    print(f"{'='*60}")

    with open(covid_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modifications = []

    # 1. Update header comment
    if content.startswith('# covid_spiral'):
        old_header = content.split('\n')[0]
        new_header = old_header.replace('covid_spiral', 'human_spiral')
        content = content.replace(old_header, new_header, 1)
        modifications.append(f"✓ Updated file header: {new_header}")

    # 2. Replace COVID genome description with Human genome
    replacements = [
        (r'SARS-CoV-2 Wuhan-Hu-1 genome', 'Human genome (GRCh38.p14)'),
        (r'viral genome', 'human genome'),
        (r'RNA-like', 'DNA-based'),
        (r'RNA-optimized geometries with viral color scheme', 'DNA-optimized geometries with human genome color scheme'),
    ]

    for old, new in replacements:
        if re.search(old, content):
            content = re.sub(old, new, content)
            modifications.append(f"✓ Replaced: '{old}' -> '{new}'")

    # 3. Replace find_covid_fasta with find_human_fasta
    find_covid = r'def find_covid_fasta\(\):(.*?)return None'
    find_human_replacement = '''def find_human_fasta():
    """Auto-detect human genome FASTA file (GRCh38.p14)"""
    possible_paths = [
        r"ncbi_dataset\\ncbi_dataset\\data\\GCF_000001405.40\\*.fna",
        r"ncbi_dataset\\data\\GCF_000001405.40\\*.fna",
        r"ncbi_dataset\\data\\*\\*.fna",
    ]
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None'''

    if re.search(find_covid, content, re.DOTALL):
        content = re.sub(find_covid, find_human_replacement, content, flags=re.DOTALL)
        modifications.append("✓ Replaced find_covid_fasta() with find_human_fasta()")

    # 4. Update load_genome function with flexible loading
    load_genome_pattern = r'def load_genome\(fasta_file\):(.*?)return genome_seq'

    enhanced_load_genome = '''def load_genome(fasta_file, max_nucleotides=None, chromosome=None):
    """
    Load genome sequence from FASTA file with flexible nucleotide limiting

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var, default 100000)
        chromosome: Specific chromosome to load (e.g., "chr1", "NC_000001.11")

    Returns:
        tuple: (sequence_string, metadata_dict)
    """
    # Check environment variable for user preference
    env_limit = os.environ.get('GENOME_LIMIT', '100000')

    if max_nucleotides is None:
        if env_limit == 'all':
            max_nucleotides = None  # Load everything
        else:
            try:
                max_nucleotides = int(env_limit)
            except ValueError:
                max_nucleotides = 100000

    print(f"Loading human genome from: {fasta_file}")

    sequences = []
    current_chr = None
    current_seq = []
    metadata = {'chromosomes': [], 'total_loaded': 0}

    with open(fasta_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous chromosome
                if current_chr and current_seq:
                    seq_str = ''.join(current_seq)
                    if chromosome is None or chromosome in current_chr:
                        sequences.append(seq_str)
                        metadata['chromosomes'].append(current_chr)

                current_chr = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())

    # Save last chromosome
    if current_chr and current_seq:
        seq_str = ''.join(current_seq)
        if chromosome is None or chromosome in current_chr:
            sequences.append(seq_str)
            metadata['chromosomes'].append(current_chr)

    # Combine all sequences
    genome_seq = ''.join(sequences)

    # Apply nucleotide limit
    if max_nucleotides and len(genome_seq) > max_nucleotides:
        genome_seq = genome_seq[:max_nucleotides]
        print(f"  Loaded {max_nucleotides:,} nucleotides from {metadata['chromosomes'][0]}")
        print(f"  (limited to {max_nucleotides:,}, set GENOME_LIMIT env var to change)")
    else:
        print(f"  Loaded {len(genome_seq):,} nucleotides from {len(metadata['chromosomes'])} chromosome(s)")

    metadata['total_loaded'] = len(genome_seq)
    return genome_seq, metadata'''

    if re.search(load_genome_pattern, content, re.DOTALL):
        content = re.sub(load_genome_pattern, enhanced_load_genome, content, flags=re.DOTALL)
        modifications.append("✓ Enhanced load_genome() with flexible loading (GENOME_LIMIT env var)")

    # 5. Update function calls
    content = content.replace('find_covid_fasta()', 'find_human_fasta()')
    content = content.replace('covid_fasta_path', 'human_fasta_path')
    modifications.append("✓ Updated function calls: find_covid_fasta() -> find_human_fasta()")

    # 6. Update load_genome call to handle tuple return
    old_load_call = r'genome_seq = load_genome\(human_fasta_path\)'
    new_load_call = 'genome_seq, metadata = load_genome(human_fasta_path)'

    if re.search(old_load_call, content):
        content = re.sub(old_load_call, new_load_call, content)
        modifications.append("✓ Updated load_genome() call to handle (sequence, metadata) tuple")

    # 7. Update print statements
    content = content.replace('SARS-CoV-2', 'Human')
    content = content.replace('COVID-19', 'Human Genome')
    modifications.append("✓ Updated print statements")

    # 8. Add encoding='utf-8' to open() if needed (for Windows compatibility)
    if "with open(fasta_file) as f:" in content:
        content = content.replace(
            "with open(fasta_file) as f:",
            "with open(fasta_file, encoding='utf-8') as f:"
        )
        modifications.append("✓ Added UTF-8 encoding to file operations")

    # Check if anything changed
    if content == original_content:
        print("⚠ No changes made - file might already be in human format or pattern not found")
        return False

    # Write the converted file
    with open(human_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ Successfully created {human_file.name}")
    print(f"\nModifications made:")
    for mod in modifications:
        print(f"  {mod}")

    return True

def main():
    """Convert all COVID spiral files to human versions"""

    script_dir = Path(__file__).parent

    # Find all covid_spiral*.py files
    covid_files = list(script_dir.glob('covid_spiral*.py'))

    if not covid_files:
        print("⚠ No covid_spiral*.py files found in current directory")
        return

    print(f"\n{'='*60}")
    print(f"BATCH CONVERSION: COVID Spiral -> Human Spiral")
    print(f"{'='*60}")
    print(f"Found {len(covid_files)} COVID spiral files to convert\n")

    converted = 0
    skipped = 0

    for covid_file in sorted(covid_files):
        # Create corresponding human filename
        human_filename = covid_file.name.replace('covid_spiral', 'human_spiral')
        human_file = script_dir / human_filename

        if human_file.exists():
            print(f"\n⚠ Skipping {covid_file.name} - {human_filename} already exists")
            skipped += 1
            continue

        if convert_to_human(covid_file, human_file):
            converted += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Converted: {converted} files")
    print(f"⚠ Skipped: {skipped} files")
    print(f"\nNew human spiral files:")
    for human_file in sorted(script_dir.glob('human_spiral*.py')):
        print(f"  • {human_file.name}")

if __name__ == '__main__':
    main()
