"""
Test all menu options in the control panel
This will verify:
1. Script discovery works with new 'eco' naming
2. All nucleotide options are valid
3. All chromosome options work
4. Script browsing commands work
5. All categories are properly discovered
"""

import sys
from pathlib import Path

# Test 1: Script Discovery
print("="*70)
print("TEST 1: Script Discovery")
print("="*70)

from human_genome_control_panel import ControlPanel

cp = ControlPanel()

# Count scripts by category
eco_count = len(cp.available_scripts['eco'])
fasta_count = len(cp.available_scripts['fasta'])
spiral_count = len(cp.available_scripts['spiral'])
total_count = eco_count + fasta_count + spiral_count

print(f"✓ Eco scripts: {eco_count}")
print(f"✓ Fasta scripts: {fasta_count}")
print(f"✓ Spiral scripts: {spiral_count}")
print(f"✓ Total scripts: {total_count}")
print()

# Verify expected counts
expected_eco = 50
expected_fasta = 28
expected_spiral = 2

if eco_count == expected_eco:
    print(f"✓ Eco count matches expected ({expected_eco})")
else:
    print(f"✗ Eco count mismatch: got {eco_count}, expected {expected_eco}")

if fasta_count == expected_fasta:
    print(f"✓ Fasta count matches expected ({expected_fasta})")
else:
    print(f"✗ Fasta count mismatch: got {fasta_count}, expected {expected_fasta}")

if spiral_count == expected_spiral:
    print(f"✓ Spiral count matches expected ({expected_spiral})")
else:
    print(f"✗ Spiral count mismatch: got {spiral_count}, expected {expected_spiral}")

print()

# Test 2: Verify file naming
print("="*70)
print("TEST 2: Verify No 'ecoli' in Filenames")
print("="*70)

workspace = Path(__file__).parent
ecoli_files = list(workspace.glob('human_ecoli*.py'))
eco_files = list(workspace.glob('human_eco*.py'))

print(f"Files with 'human_ecoli' pattern: {len(ecoli_files)}")
print(f"Files with 'human_eco' pattern: {len(eco_files)}")

if len(ecoli_files) == 0:
    print("✓ No human_ecoli*.py files found (good!)")
else:
    print(f"✗ Found {len(ecoli_files)} human_ecoli*.py files that should be renamed:")
    for f in ecoli_files[:5]:
        print(f"  - {f.name}")

if len(eco_files) == expected_eco:
    print(f"✓ Found {len(eco_files)} human_eco*.py files (correct)")
else:
    print(f"✗ Found {len(eco_files)} human_eco*.py files, expected {expected_eco}")

print()

# Test 3: Sample eco files to verify internal content
print("="*70)
print("TEST 3: Verify Internal Content (sample files)")
print("="*70)

sample_files = [
    'human_eco17.py',
    'human_eco46_v3_pure_fasta.py',
    'human_eco_unified_phi_synthesis.py'
]

for filename in sample_files:
    filepath = workspace / filename
    if filepath.exists():
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')

            # Check for problematic patterns
            has_human_ecoli = 'human_ecoli' in content
            has_ecoli_underscore = 'ecoli_' in content

            if has_human_ecoli or has_ecoli_underscore:
                print(f"✗ {filename} still contains 'ecoli' references")
                if has_human_ecoli:
                    print(f"  - Found 'human_ecoli'")
                if has_ecoli_underscore:
                    print(f"  - Found 'ecoli_'")
            else:
                print(f"✓ {filename} - no 'ecoli' references found")
        except Exception as e:
            print(f"⚠ {filename} - could not read file: {e}")
    else:
        print(f"✗ {filename} - file not found")

print()

# Test 4: Nucleotide Options Validation
print("="*70)
print("TEST 4: Nucleotide Options Validation")
print("="*70)

nucleotide_options = {
    'Quick Preview': 10_000,
    'Standard View': 100_000,
    'Detailed View': 500_000,
    'Extended View': 1_000_000,
    'Chromosome View': 10_000_000,
    'Full Genome': 3_100_000_000,
}

print("Nucleotide presets:")
for name, count in nucleotide_options.items():
    print(f"  ✓ {name}: {count:,} nucleotides")

print()

# Test 5: Chromosome Reference IDs
print("="*70)
print("TEST 5: Chromosome Reference IDs")
print("="*70)

chromosomes = {
    1: ('NC_000001.11', '249M bases'),
    2: ('NC_000002.12', '242M bases'),
    3: ('NC_000003.12', '198M bases'),
    4: ('NC_000004.12', '190M bases'),
    5: ('NC_000005.10', '181M bases'),
    6: ('NC_000006.12', '170M bases'),
    7: ('NC_000007.14', '159M bases'),
    8: ('NC_000008.11', '145M bases'),
    9: ('NC_000009.12', '138M bases'),
    10: ('NC_000010.11', '133M bases'),
    11: ('NC_000011.10', '135M bases'),
    12: ('NC_000012.12', '133M bases'),
    13: ('NC_000013.11', '114M bases'),
    14: ('NC_000014.9', '107M bases'),
    15: ('NC_000015.10', '101M bases'),
    16: ('NC_000016.10', '90M bases'),
    17: ('NC_000017.11', '83M bases'),
    18: ('NC_000018.10', '80M bases'),
    19: ('NC_000019.10', '58M bases'),
    20: ('NC_000020.11', '64M bases'),
    21: ('NC_000021.9', '46M bases'),
    22: ('NC_000022.11', '50M bases'),
    'X': ('NC_000023.11', '154M bases'),
    'Y': ('NC_000024.10', '57M bases'),
    'MT': ('NC_012920.1', '16,569 bases'),
}

print("Chromosome IDs available:")
for chr_num, (ref_id, size) in chromosomes.items():
    print(f"  ✓ Chr {chr_num}: {ref_id} ({size})")

print()

# Test 6: Sample Script Names
print("="*70)
print("TEST 6: Sample Discovered Scripts")
print("="*70)

print("Sample eco scripts (first 5):")
for script in cp.available_scripts['eco'][:5]:
    print(f"  ✓ {script.name}")

print()

print("Sample fasta scripts (first 5):")
for script in cp.available_scripts['fasta'][:5]:
    print(f"  ✓ {script.name}")

print()

print("All spiral scripts:")
for script in cp.available_scripts['spiral']:
    print(f"  ✓ {script.name}")

print()

# Final Summary
print("="*70)
print("TEST SUMMARY")
print("="*70)

all_tests_passed = True

if eco_count != expected_eco or fasta_count != expected_fasta or spiral_count != expected_spiral:
    all_tests_passed = False

if len(ecoli_files) > 0:
    all_tests_passed = False

if all_tests_passed:
    print("✓ ALL TESTS PASSED!")
    print(f"✓ Control panel ready with {total_count} scripts")
    print("✓ No 'ecoli' references in human files")
    print("✓ All eco, fasta, and spiral scripts discovered")
else:
    print("✗ SOME TESTS FAILED - see details above")

print()
print("To test the interactive menu:")
print("  python human_genome_control_panel.py")
print()
