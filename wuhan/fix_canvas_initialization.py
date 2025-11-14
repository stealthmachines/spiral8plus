#!/usr/bin/env python3
"""
Master fix script: Add missing canvas/view/genome initialization to ALL human_eco files.
This script scans for common patterns and adds the necessary code.
"""
import os
import re

workspace = r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8"

CANVAS_INIT_TEMPLATE = """
# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
"""

GENOME_LOAD_TEMPLATE = """
# Load genome
fasta_file = find_human_fasta()
genome_seq = load_genome(fasta_file)
genome_len = len(genome_seq)
"""

def needs_genome_loading(content):
    """Check if file needs genome loading code"""
    return ('genome_seq[' in content or 'genome_len' in content) and 'genome_seq = ' not in content

def needs_canvas_init(content):
    """Check if file needs canvas initialization"""
    return 'canvas.show()' in content and 'canvas = scene.SceneCanvas' not in content

def find_insertion_point_for_genome(content):
    """Find where to insert genome loading (after load_genome function)"""
    match = re.search(r'(\n\s*return sequence\s*\n)', content)
    if match:
        return match.end()
    return None

def find_insertion_point_for_canvas(content):
    """Find where to insert canvas (after genome loading or before def update)"""
    # Try after genome_len =
    match = re.search(r'(genome_len\s*=\s*[^\n]+\n)', content)
    if match:
        return match.end()

    # Try before def update
    match = re.search(r'\n(def update\()', content)
    if match:
        return match.start() + 1

    return None

def extract_needed_variables(content):
    """Extract variable names used but not defined"""
    # Common variables in these scripts
    variables = {
        'nucleoid_radius': '12.0',
        'cell_radius': '25.0',
        'total_twist': '4 * np.pi',
        'core_radius': '15.0',
        'strand_sep': '0.5',
        'twist_factor': '2 * np.pi',
        'frame': '0'
    }

    needed = []
    for var, default in variables.items():
        if var in content and f'{var} =' not in content[:content.find(var)]:
            needed.append((var, default))

    return needed

def needs_strand_init(content):
    """Check if strand1/strand2 need initialization"""
    return 'strand1.set_data' in content and 'strand1 = Line' not in content

def fix_file(filepath, filename):
    """Apply all necessary fixes to a file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False, "Can't read"

    original_content = content
    fixes = []

    # Fix 1: Add genome loading if needed
    if needs_genome_loading(content):
        insert_point = find_insertion_point_for_genome(content)
        if insert_point:
            content = content[:insert_point] + GENOME_LOAD_TEMPLATE + content[insert_point:]
            fixes.append("genome loading")

    # Fix 2: Add canvas initialization if needed
    if needs_canvas_init(content):
        insert_point = find_insertion_point_for_canvas(content)
        if insert_point:
            # Check if we need additional variables
            needed_vars = extract_needed_variables(content)
            var_init = "\n# Variables\n"
            for var, default in needed_vars:
                var_init += f"{var} = {default}\n"

            # Check if we need strand initialization
            strand_init = ""
            if needs_strand_init(content):
                strand_init = "\n# Strands\nstrand1 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=view.scene)\nstrand2 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=view.scene)\n"

            # Add common list initializations if they appear in code
            list_vars = []
            for var in ['rungs', 'echoes', 'links', 'labels', 'centers', 'emerged', 'membrane', 'organelles']:
                if var in content and f'{var} = ' not in content[:insert_point]:
                    list_vars.append(var)

            list_init = ""
            if list_vars:
                list_init = f"\n{', '.join(list_vars)} = " + ", ".join(["[]"] * len(list_vars)) + "\n"

            full_init = CANVAS_INIT_TEMPLATE + strand_init + var_init + list_init
            content = content[:insert_point] + full_init + content[insert_point:]
            fixes.append("canvas/view/variables")

    # Save if modified
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, ", ".join(fixes)

    return False, "No changes needed"

def main():
    """Process all human_eco files"""
    fixed_count = 0

    print("Fixing missing canvas/genome initialization...\n")

    for filename in sorted(os.listdir(workspace)):
        if not (filename.endswith('.py') and filename.startswith('human_eco')):
            continue

        filepath = os.path.join(workspace, filename)
        modified, message = fix_file(filepath, filename)

        if modified:
            print(f"✓ {filename}: {message}")
            fixed_count += 1

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
