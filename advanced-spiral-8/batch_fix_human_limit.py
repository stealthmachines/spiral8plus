"""
Batch fix: Add nucleotide limit to human genome files
Modifies load_genome function to only load a limited number of nucleotides
to make visualization practical with the huge human genome
"""

import os
import re
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Default limit: 100,000 nucleotides (reasonable for visualization)
DEFAULT_LIMIT = 100000

# New load_genome function with limit
NEW_LOAD_GENOME = '''def load_genome(fasta_file, max_nucleotides=100000):
    """Load genome sequence with nucleotide limit for large genomes"""
    seq = []
    count = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            bases = list(line.strip().upper())
            remaining = max_nucleotides - count
            if remaining <= 0:
                break
            seq.extend(bases[:remaining])
            count += len(bases[:remaining])
    print(f"Loaded {len(seq):,} nucleotides from {os.path.basename(fasta_file)}")
    return seq'''

def fix_file(filepath):
    """Add nucleotide limit to load_genome function"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already fixed
    if 'max_nucleotides' in content:
        print(f"  ⚠ Already fixed")
        return False

    # Pattern 1: Simple load_genome function (most common)
    pattern1 = r'def load_genome\(fasta_file\):\s*seq = \[\]\s*with open\(fasta_file\) as f:\s*for line in f:\s*if line\.startswith\(">"[^\)]*\):\s*continue\s*seq\.extend\(list\(line\.strip\(\)\.upper\(\)\)\)\s*return seq'

    if re.search(pattern1, content, re.DOTALL):
        content = re.sub(pattern1, NEW_LOAD_GENOME, content, flags=re.DOTALL)
        print(f"  ✓ Fixed (pattern 1)")
    else:
        # Pattern 2: load_genome with variable assignment
        pattern2 = r'def load_genome\(fasta_file\):\s*"""[^"]*"""\s*seq = \[\]\s*with open\(fasta_file\) as f:\s*for line in f:\s*if line\.startswith\(">"[^\)]*\):\s*continue\s*seq\.extend\(list\(line\.strip\(\)\.upper\(\)\)\)\s*return seq'

        if re.search(pattern2, content, re.DOTALL):
            content = re.sub(pattern2, NEW_LOAD_GENOME, content, flags=re.DOTALL)
            print(f"  ✓ Fixed (pattern 2)")
        else:
            # Pattern 3: Any def load_genome up to return seq
            pattern3 = r'def load_genome\([^)]*\):.*?return seq'
            if re.search(pattern3, content, re.DOTALL):
                content = re.sub(pattern3, NEW_LOAD_GENOME, content, flags=re.DOTALL, count=1)
                print(f"  ✓ Fixed (generic pattern)")
            else:
                print(f"  ⚠ Could not find load_genome function")
                return False

    # Add import os if not present (for basename)
    if 'import os' not in content:
        # Find first import and add os
        import_match = re.search(r'^(import [^\n]+)', content, re.MULTILINE)
        if import_match:
            content = content.replace(import_match.group(0), import_match.group(0) + '\nimport os', 1)
        else:
            content = 'import os\n' + content

    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return True

def main():
    """Main fixing process"""
    print("\n" + "="*70)
    print("BATCH FIX: Add Nucleotide Limit to Human Genome Files")
    print("="*70)
    print(f"Limiting to {DEFAULT_LIMIT:,} nucleotides per file")
    print("="*70 + "\n")

    # Find all human_*.py files
    import glob
    all_files = sorted(glob.glob('human_*.py'))

    if not all_files:
        print("⚠ No human_*.py files found!")
        return

    print(f"Found {len(all_files)} files to fix\n")

    fixed = []
    skipped = []
    failed = []

    for filepath in all_files:
        print(f"Fixing: {filepath}")
        try:
            result = fix_file(filepath)
            if result:
                fixed.append(filepath)
            else:
                skipped.append(filepath)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append(filepath)
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Fixed: {len(fixed)} files")
    print(f"⚠ Skipped: {len(skipped)} files")
    print(f"✗ Failed: {len(failed)} files")

    if fixed:
        print(f"\nFixed {len(fixed)} files to load max {DEFAULT_LIMIT:,} nucleotides")
        print("This makes human genome visualization practical!")

    print("="*70 + "\n")

if __name__ == '__main__':
    main()
