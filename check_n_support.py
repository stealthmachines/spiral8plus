#!/usr/bin/env python3
"""Check which files actually have 'N' support in their base_colors"""
import os
import re

# Get workspace root
workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan"
script_dir = os.path.join(workspace, "advanced-spiral-8")

# Find all human_eco*.py files
files = [f for f in os.listdir(script_dir) if f.startswith('human_eco') and f.endswith('.py')]
files.sort()

print(f"Checking {len(files)} files for 'N' support...\n")

has_n = []
missing_n = []

for filename in files:
    filepath = os.path.join(script_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if file has 'N' in base_colors
    if "'N':" in content or '"N":' in content:
        has_n.append(filename)
        print(f"✓ {filename}")
    else:
        missing_n.append(filename)
        print(f"✗ {filename}")

print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Files with 'N' support: {len(has_n)}/{len(files)}")
print(f"  Files missing 'N': {len(missing_n)}/{len(files)}")

if len(missing_n) <= 10:
    print(f"\nMissing 'N':")
    for f in missing_n:
        print(f"  - {f}")
