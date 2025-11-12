"""Fix all human genome scripts to properly handle GENOME_LIMIT='all' for full genome loading"""

import os
import re
from pathlib import Path

def fix_genome_limit_handling(filepath):
    """Fix a single script to handle GENOME_LIMIT='all' properly"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern 1: Scripts with load_genome function that have max_nucleotides parameter
    # Replace the ValueError fallback to check for 'all' first
    pattern1 = r'(\s+if max_nucleotides is None:\s+env_limit = os\.environ\.get\(\'GENOME_LIMIT\', \'100000\'\)\s+try:\s+max_nucleotides = int\(env_limit\)\s+except ValueError:\s+max_nucleotides = 100000)'

    replacement1 = r'''\1
        # FIXED: Handle 'all' for full genome
        if max_nucleotides is None:
            env_limit = os.environ.get('GENOME_LIMIT', '100000')
            if env_limit == 'all':
                max_nucleotides = None  # Load full genome
            else:
                try:
                    max_nucleotides = int(env_limit)
                except ValueError:
                    max_nucleotides = 100000'''

    # Actually, let me do a more targeted fix - replace the exact block
    old_block = """    # Get from environment if not specified
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        try:
            max_nucleotides = int(env_limit)
        except ValueError:
            max_nucleotides = 100000"""

    new_block = """    # Get from environment if not specified
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        if env_limit == 'all':
            max_nucleotides = None  # Load full genome
        else:
            try:
                max_nucleotides = int(env_limit)
            except ValueError:
                max_nucleotides = 100000"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "Fixed load_genome ValueError handling"

    # Pattern 2: Scripts that directly read into max_nucleotides variable (no function)
    # Example: max_nucleotides = int(os.environ.get('GENOME_LIMIT', '100000'))
    old_direct = "max_nucleotides = int(os.environ.get('GENOME_LIMIT', '100000'))"
    new_direct = """env_limit = os.environ.get('GENOME_LIMIT', '100000')
    max_nucleotides = None if env_limit == 'all' else int(env_limit)"""

    if old_direct in content:
        content = content.replace(old_direct, new_direct)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "Fixed direct max_nucleotides assignment"

    # Pattern 3: Scripts with genome_limit variable instead of max_nucleotides
    old_genome_limit = "genome_limit = os.environ.get('GENOME_LIMIT', '100000')"
    new_genome_limit = """env_limit_str = os.environ.get('GENOME_LIMIT', '100000')
    genome_limit = None if env_limit_str == 'all' else int(env_limit_str)"""

    if old_genome_limit in content:
        content = content.replace(old_genome_limit, new_genome_limit)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "Fixed genome_limit handling"

    return False, "No matching pattern found"

def main():
    workspace = Path(r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8")

    # Find all human_*.py scripts
    scripts = sorted(workspace.glob("human_*.py"))

    # Exclude the control panel itself
    scripts = [s for s in scripts if s.name != "human_genome_control_panel.py"]

    print(f"Found {len(scripts)} human genome scripts to fix")
    print("="*70)

    fixed = 0
    skipped = 0

    for script in scripts:
        success, message = fix_genome_limit_handling(script)
        if success:
            print(f"✓ {script.name:40} - {message}")
            fixed += 1
        else:
            print(f"○ {script.name:40} - {message}")
            skipped += 1

    print("="*70)
    print(f"\nResults:")
    print(f"  ✓ Fixed: {fixed}")
    print(f"  ○ Skipped: {skipped}")
    print(f"  Total: {len(scripts)}")
    print()
    print("All scripts now support GENOME_LIMIT='all' for full genome loading!")

if __name__ == "__main__":
    main()
