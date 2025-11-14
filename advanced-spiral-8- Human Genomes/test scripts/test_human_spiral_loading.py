"""Quick test of human spiral files' genome loading"""
import os
import glob

os.environ['GENOME_LIMIT'] = '50000'

def find_human_fasta():
    """Auto-detect human genome FASTA file (GRCh38.p14)"""
    possible_paths = [
        r"ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\data\*\*.fna",
    ]
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def load_genome(fasta_file, max_nucleotides=None, chromosome=None):
    """Load genome sequence from FASTA file with flexible nucleotide limiting"""
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

    seq = []
    current_chr = None
    metadata = {'chromosomes': [], 'total_loaded': 0}

    print(f"Loading Human genome from: {fasta_file}")
    with open(fasta_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith(">"):
                chr_name = line.strip()[1:].split()[0]
                if current_chr is None:
                    current_chr = chr_name
                    metadata['chromosomes'].append(chr_name)
                    print(f"Sequence: {line.strip()}")
                elif chromosome is None or chromosome in chr_name:
                    # Loading multiple chromosomes
                    current_chr = chr_name
                    metadata['chromosomes'].append(chr_name)
                continue

            if chromosome is None or (current_chr and chromosome in current_chr):
                seq.extend(list(line.strip().upper()))

                # Check if we've reached the limit
                if max_nucleotides and len(seq) >= max_nucleotides:
                    seq = seq[:max_nucleotides]
                    metadata['total_loaded'] = len(seq)
                    print(f"  Loaded {max_nucleotides:,} nucleotides from {metadata['chromosomes'][0]}")
                    print(f"  (limited to {max_nucleotides:,}, set GENOME_LIMIT env var to change)")
                    return seq, metadata

    metadata['total_loaded'] = len(seq)
    if max_nucleotides is None:
        print(f"  Loaded {len(seq):,} nucleotides from {len(metadata['chromosomes'])} chromosome(s)")

    return seq, metadata

# Test
fasta_path = find_human_fasta()
print(f"\nFound FASTA: {fasta_path}")

genome_seq, metadata = load_genome(fasta_path)
print(f"\n✅ Loaded {len(genome_seq):,} nucleotides")
print(f"   Chromosomes: {metadata['chromosomes']}")
print(f"   Total loaded: {metadata['total_loaded']:,}")
