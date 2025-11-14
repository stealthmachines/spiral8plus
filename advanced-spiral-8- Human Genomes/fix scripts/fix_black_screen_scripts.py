"""
Fix black screen issues in visualization scripts.

Problem: Scripts reference visual objects in update() but never initialize them,
causing black screens or NameErrors.

Solution: Add visual object initialization after canvas setup and before timer starts.
"""

import os
import re
from pathlib import Path

def find_missing_visuals(content):
    """Find visual objects used in update() but not initialized"""
    # Find all visual object references in update function
    update_match = re.search(r'def update\(.*?\):(.*?)(?=\ndef |\nif __name__|timer\s*=|\Z)', content, re.DOTALL)
    if not update_match:
        return []

    update_code = update_match.group(1)

    # Common visual object patterns
    patterns = {
        'comp_line1': r'comp_line1\.set_data|comp_line1\.update',
        'comp_line2': r'comp_line2\.set_data|comp_line2\.update',
        'res_markers': r'res_markers\.set_data|res_markers\.update',
        'pct_text': r'pct_text\.text\s*=',
        'percent_text': r'percent_text\.text\s*=',
        'line1': r'\bline1\.set_data|line1\.update',
        'line2': r'\bline2\.set_data|line2\.update',
        'strand1': r'strand1\.set_data|strand1\.update',
        'strand2': r'strand2\.set_data|strand2\.update',
        'strand_vis_1': r'strand_vis_1\.set_data|strand_vis_1\.update',
        'strand_vis_2': r'strand_vis_2\.set_data|strand_vis_2\.update',
        'vis_strand_A': r'vis_strand_A\.set_data|vis_strand_A\.update',
        'vis_strand_B': r'vis_strand_B\.set_data|vis_strand_B\.update',
        'core_line1': r'core_line1\.set_data|core_line1\.update',
        'core_line2': r'core_line2\.set_data|core_line2\.update',
        'helix1': r'helix1\.set_data|helix1\.update',
        'helix2': r'helix2\.set_data|helix2\.update',
        'main_strand1': r'main_strand1\.set_data|main_strand1\.update',
        'main_strand2': r'main_strand2\.set_data|main_strand2\.update',
        'strand1_vis': r'strand1_vis\.set_data|strand1_vis\.update',
        'strand2_vis': r'strand2_vis\.set_data|strand2_vis\.update',
        'donut_vis': r'donut_vis\.set_data|donut_vis\.update',
        'org_marker': r"org\['marker'\]\.set_data|org_marker\.set_data",
    }

    missing = []
    for var_name, pattern in patterns.items():
        if re.search(pattern, update_code):
            # Check if variable is initialized before update()
            init_pattern = rf'\b{var_name}\s*='
            if not re.search(init_pattern, content[:content.find('def update')]):
                missing.append(var_name)

    return missing

def generate_initialization_code(missing_visuals, content):
    """Generate initialization code for missing visual objects"""
    code_lines = []

    # Determine view name (comp_view or view)
    if 'comp_view' in content:
        view_name = 'comp_view'
    else:
        view_name = 'view'

    for visual in missing_visuals:
        if 'line' in visual or 'strand' in visual:
            code_lines.append(f"{visual} = Line(pos=np.zeros((100, 3)), color='cyan', width=2, parent={view_name}.scene)")
        elif 'markers' in visual:
            code_lines.append(f"{visual} = Markers(pos=np.zeros((100, 3)), face_color='white', size=6, parent={view_name}.scene)")
        elif 'text' in visual:
            code_lines.append(f"{visual} = Text('0%', pos=(0, 0, 10), color='white', font_size=14, parent={view_name}.scene)")

    return '\n'.join(code_lines)

def fix_script(script_path):
    """Fix a single script by adding missing visual object initializations"""
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find missing visuals
    missing = find_missing_visuals(content)
    if not missing:
        return None, "No missing visuals found"

    # Find where to insert initialization (after camera setup, before timer)
    # Look for camera = 'turntable' or camera.distance
    camera_match = re.search(r"(comp_view|view)\.camera\s*=\s*'turntable'", content)
    if not camera_match:
        camera_match = re.search(r"(comp_view|view)\.camera\.distance", content)

    if not camera_match:
        return None, "Could not find camera setup location"

    insert_pos = camera_match.end()

    # Find the end of current line
    next_newline = content.find('\n', insert_pos)
    if next_newline == -1:
        return None, "Could not find insertion point"

    insert_pos = next_newline + 1

    # Generate initialization code
    init_code = generate_initialization_code(missing, content)

    # Insert the code
    new_content = content[:insert_pos] + '\n# Initialize visual objects\n' + init_code + '\n' + content[insert_pos:]

    return new_content, f"Added initialization for: {', '.join(missing)}"

def main():
    """Fix all problematic scripts"""
    # Scripts with black screen issues
    problem_scripts = [
        'human_eco25.py', 'human_eco28.py', 'human_eco29.py', 'human_eco30.py',
        'human_eco32.py', 'human_eco34.py', 'human_eco37.py', 'human_eco38.py',
        'human_fasta2.py', 'human_fasta3.py', 'human_fasta4.py', 'human_fasta4b.py',
        'human_fasta4c.py', 'human_fasta4d.py', 'human_fasta4e.py', 'human_fasta4f.py',
        'human_fasta4g.py', 'human_fasta5.py', 'human_fasta6.py', 'human_fasta7.py',
        'human_fasta8.py', 'human_fasta9.py'
    ]

    current_dir = Path(__file__).parent
    fixed_count = 0
    error_count = 0

    print(f"\n{'='*70}")
    print(f"Fixing Black Screen Scripts")
    print(f"{'='*70}\n")

    for script_name in problem_scripts:
        script_path = current_dir / script_name
        if not script_path.exists():
            print(f"[SKIP] {script_name} - File not found")
            continue

        print(f"Processing: {script_name}")
        new_content, message = fix_script(script_path)

        if new_content:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ FIXED: {message}")
            fixed_count += 1
        else:
            print(f"  ⚠️  SKIP: {message}")
            error_count += 1

    print(f"\n{'='*70}")
    print(f"Summary: {fixed_count} fixed, {error_count} skipped")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
