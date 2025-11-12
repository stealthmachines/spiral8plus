"""Direct test of human_spiral8.py and human_spiral9.py genome loading"""

import os
import glob
import sys

# Set small limit for testing
os.environ['GENOME_LIMIT'] = '5000'

print("="*70)
print("DIRECT GENOME LOADING TEST")
print("="*70)

# Test human_spiral8.py
print("\n1. Testing human_spiral8.py")
print("-"*70)

exec(open('human_spiral8.py', 'r', encoding='utf-8').read().split('# Load the coronavirus genome')[0])

try:
    fasta_path = find_human_fasta()
    print(f"✅ Found FASTA: {fasta_path}")

    genome_seq, metadata = load_genome(fasta_path)
    print(f"✅ Loaded {len(genome_seq):,} nucleotides")
    print(f"✅ Metadata: {metadata}")
    print("✅ human_spiral8.py: PASSED")
except Exception as e:
    print(f"❌ human_spiral8.py: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test human_spiral9.py
print("\n2. Testing human_spiral9.py")
print("-"*70)

exec(open('human_spiral9.py', 'r', encoding='utf-8').read().split('# Load the coronavirus genome')[0])

try:
    fasta_path = find_human_fasta()
    print(f"✅ Found FASTA: {fasta_path}")

    genome_seq, metadata = load_genome(fasta_path)
    print(f"✅ Loaded {len(genome_seq):,} nucleotides")
    print(f"✅ Metadata: {metadata}")
    print("✅ human_spiral9.py: PASSED")
except Exception as e:
    print(f"❌ human_spiral9.py: FAILED - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETED")
print("="*70)
