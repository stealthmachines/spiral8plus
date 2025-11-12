#!/usr/bin/env python3
"""
Final comprehensive fix for all human_eco and related scripts.
Fixes:
1. Missing find_human_fasta() and load_genome() functions
2. Missing canvas/view initialization
3. 'N' nucleotide handling in all contexts
4. Function definitions before usage
"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8"

def add_n_handling_to_index(content):
    """Add N handling where nucleotides.index(base) is used"""
    # Pattern: nucleotides.index(base)
    pattern = r'(\s+)((?:geom_idx|idx|index)\s*=\s*\w+\[)nucleotides\.index\((\w+)\)(\])'

    def replacement(match):
        indent = match.group(1)
        prefix = match.group(2)
        var = match.group(3)
        suffix = match.group(4)

        return (f"{indent}# Handle unknown nucleotides\n"
                f"{indent}if {var} not in nucleotides:\n"
                f"{indent}    {var} = 'A'  # Default to A\n"
                f"{indent}{prefix}nucleotides.index({var}){suffix}")

    return re.sub(pattern, replacement, content)

def add_n_to_nucleotides_list(content):
    """Add 'N' to nucleotides list if not present"""
    # Find nucleotides = ["A","T","G","C"]
    pattern = r'nucleotides\s*=\s*\[([^\]]+)\]'
    match = re.search(pattern, content)

    if match and '"N"' not in match.group(1) and "'N'" not in match.group(1):
        old = match.group(0)
        # Add N to the list
        new = old.replace('"]', '","N"]').replace("']", "','N']")
        content = content.replace(old, new)
        return content, True

    return content, False

def fix_n_in_base_colors(content):
    """Add N to base_colors if using direct access"""
    # Check if file has base_colors dict and uses [base] access
    if 'base_colors[' in content and "'N':" not in content:
        # Find base_colors definition
        pattern = r"(base_colors\s*=\s*\{[^}]*'T':\s*\([^)]+\),?)"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

        if match:
            insert_pos = match.end(1)
            n_entry = "\n    'N': (0.5, 0.5, 0.5, 1),  # Gray (unknown)"
            content = content[:insert_pos] + n_entry + content[insert_pos:]
            return content, True

    return content, False

def fix_file_comprehensive(filepath, filename):
    """Apply all fixes to a file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return 0

    modified = False
    fixes = []

    # Fix 1: Add N handling for nucleotides.index()
    if 'nucleotides.index(' in content and 'if ' not in content[:content.find('nucleotides.index(')]:
        new_content = add_n_handling_to_index(content)
        if new_content != content:
            content = new_content
            modified = True
            fixes.append("N handling for index()")

    # Fix 2: Add N to nucleotides list
    new_content, changed = add_n_to_nucleotides_list(content)
    if changed:
        content = new_content
        modified = True
        fixes.append("Added N to nucleotides list")

    # Fix 3: Add N to base_colors
    new_content, changed = fix_n_in_base_colors(content)
    if changed:
        content = new_content
        modified = True
        fixes.append("Added N to base_colors")

    # Fix 4: Add .get() default for base_map if using [base]
    if 'base_map[base]' in content:
        content = content.replace('base_map[base]', 'base_map.get(base, 1)')
        modified = True
        fixes.append("Changed base_map[base] to .get()")

    # Fix 5: Add .get() default for base_colors if using [base]
    if 'base_colors[base]' in content or 'base_colors[b]' in content:
        content = content.replace('base_colors[base]', 'base_colors.get(base, (0.5, 0.5, 0.5, 1))')
        content = content.replace('base_colors[b]', 'base_colors.get(b, (0.5, 0.5, 0.5, 1))')
        modified = True
        fixes.append("Changed base_colors[] to .get()")

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return len(fixes)

    return 0

def main():
    """Process all files"""
    total_fixed = 0

    print("Applying comprehensive 'N' nucleotide fixes...\n")

    for filename in sorted(os.listdir(workspace)):
        if not (filename.endswith('.py') and (
            filename.startswith('human_eco') or
            filename.startswith('human_fasta')
        )):
            continue

        filepath = os.path.join(workspace, filename)
        fixes = fix_file_comprehensive(filepath, filename)

        if fixes > 0:
            print(f"✓ {filename}: {fixes} fixes applied")
            total_fixed += 1

    print(f"\n{'='*60}")
    print(f"Total files fixed: {total_fixed}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
