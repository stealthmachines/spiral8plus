#!/usr/bin/env python3
"""
FINAL VALIDATION: Determine which scripts are actually production-ready for human genome.

Tests:
1. Has 'N' support OR uses .get() with defaults (safe from KeyError)
2. Has load_genome() function (GENOME_LIMIT support)
3. Doesn't use SeqIO.read() (multi-chromosome safe)
"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan"

def check_file(filepath):
    """Comprehensive file check"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return None

    # Check 1: 'N' nucleotide safety
    has_n_explicit = "'N':" in content or '"N":' in content
    uses_get = '.get(' in content and 'base' in content
    n_safe = has_n_explicit or uses_get

    # Check 2: GENOME_LIMIT support
    has_load_genome = 'def load_genome(' in content
    has_genome_limit = 'GENOME_LIMIT' in content
    limit_support = has_load_genome or has_genome_limit

    # Check 3: Multi-chromosome safe
    uses_seqio_read = 'SeqIO.read(' in content
    multi_chr_safe = not uses_seqio_read

    # Overall production ready
    production_ready = n_safe and limit_support and multi_chr_safe

    return {
        'n_safe': n_safe,
        'limit_support': limit_support,
        'multi_chr_safe': multi_chr_safe,
        'production_ready': production_ready,
        'has_n_explicit': has_n_explicit,
        'uses_get': uses_get
    }

def main():
    """Check all scripts"""

    # Get all script files
    all_files = []

    # Workspace root
    for f in os.listdir(workspace):
        if f.endswith('.py') and (f.startswith('eco') or f.startswith('fasta') or f.startswith('spiral')):
            all_files.append(f)

    # advanced-spiral-8
    spiral_dir = os.path.join(workspace, 'advanced-spiral-8')
    if os.path.exists(spiral_dir):
        for f in os.listdir(spiral_dir):
            if f.endswith('.py') and f.startswith('human_eco'):
                all_files.append(f'advanced-spiral-8/{f}')

    all_files.sort()

    # Stats
    total = 0
    n_safe_count = 0
    limit_count = 0
    multi_chr_count = 0
    production_ready_count = 0
    explicit_n_count = 0

    not_ready = []

    for filename in all_files:
        total += 1
        if '/' in filename:
            filepath = os.path.join(workspace, filename.replace('/', '\\'))
        else:
            filepath = os.path.join(workspace, filename)

        result = check_file(filepath)
        if not result:
            continue

        if result['n_safe']:
            n_safe_count += 1
        if result['limit_support']:
            limit_count += 1
        if result['multi_chr_safe']:
            multi_chr_count += 1
        if result['production_ready']:
            production_ready_count += 1
        if result['has_n_explicit']:
            explicit_n_count += 1

        if not result['production_ready']:
            not_ready.append((filename, result))

    print(f"{'='*70}")
    print(f"PRODUCTION READINESS REPORT")
    print(f"{'='*70}\n")

    print(f"Total scripts analyzed: {total}\n")

    print(f"Safety Checks:")
    print(f"  ✓ 'N' safe (explicit OR .get()): {n_safe_count}/{total} ({n_safe_count/total*100:.1f}%)")
    print(f"    - Explicit 'N' entries: {explicit_n_count}")
    print(f"    - Using .get() safely: {n_safe_count - explicit_n_count}")
    print(f"  ✓ GENOME_LIMIT support: {limit_count}/{total} ({limit_count/total*100:.1f}%)")
    print(f"  ✓ Multi-chromosome safe: {multi_chr_count}/{total} ({multi_chr_count/total*100:.1f}%)")
    print(f"\n{'='*70}")
    print(f"PRODUCTION READY: {production_ready_count}/{total} ({production_ready_count/total*100:.1f}%)")
    print(f"{'='*70}\n")

    if not_ready:
        print(f"Scripts NOT production ready ({len(not_ready)}):\n")
        for filename, result in not_ready[:15]:
            issues = []
            if not result['n_safe']:
                issues.append("'N' unsafe")
            if not result['limit_support']:
                issues.append("No GENOME_LIMIT")
            if not result['multi_chr_safe']:
                issues.append("SeqIO.read()")

            print(f"  ⚠ {filename}")
            print(f"     Issues: {', '.join(issues)}")

        if len(not_ready) > 15:
            print(f"\n  ... and {len(not_ready)-15} more")
    else:
        print(f"✓ ALL SCRIPTS PRODUCTION READY!")

if __name__ == '__main__':
    main()
