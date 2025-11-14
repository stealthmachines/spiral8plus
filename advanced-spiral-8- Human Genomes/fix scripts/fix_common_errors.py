"""
Fix common issues in all human genome visualization scripts:
1. Add 'N' to base_colors dictionary (for unknown nucleotides)
2. Fix 'canvas' NameError by ensuring it's defined
"""

import re
from pathlib import Path

def fix_base_colors(content):
    """Add 'N' to base_colors dictionary"""
    # Find base_colors dictionary
    old_pattern = r"base_colors = \{[^}]*'T': \([^)]+\),[^\}]*\}"

    # Check if N is already in there
    if "'N':" in content:
        return content, False

    # Replace with version that includes N
    new_colors = """base_colors = {
    'A': (1, 0, 0, 1),   # Red
    'C': (0, 1, 0, 1),   # Green
    'G': (0, 0, 1, 1),   # Blue
    'T': (1, 1, 0, 1),   # Yellow
    'N': (0.5, 0.5, 0.5, 1),  # Gray (unknown)
}"""

    content = re.sub(old_pattern, new_colors, content, flags=re.DOTALL)
    return content, True

def fix_canvas_error(content):
    """Fix NameError for undefined 'canvas' variable"""
    # Look for canvas.show() without canvas being defined
    if 'canvas.show()' in content and 'canvas = scene.SceneCanvas' not in content:
        # Replace 'view = canvas.central_widget' pattern
        old_pattern = r'view = canvas\.central_widget\.add_view\(\)'
        new_code = '''canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()'''

        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_code, content)
            return content, True

    return content, False

def process_file(filepath):
    """Process a single file"""
    print(f"Processing {filepath.name}...")

    content = filepath.read_text(encoding='utf-8', errors='ignore')

    colors_fixed = False
    canvas_fixed = False

    # Fix base_colors
    content, colors_fixed = fix_base_colors(content)

    # Fix canvas error
    content, canvas_fixed = fix_canvas_error(content)

    if colors_fixed or canvas_fixed:
        filepath.write_text(content, encoding='utf-8')
        fixes = []
        if colors_fixed:
            fixes.append("Added 'N' to base_colors")
        if canvas_fixed:
            fixes.append("Fixed canvas definition")
        print(f"  ✓ {', '.join(fixes)}")
        return True
    else:
        print(f"  ⊘ No fixes needed")
        return False

def main():
    workspace = Path(__file__).parent

    # Get all human scripts
    eco_files = sorted(workspace.glob('human_eco*.py'))
    fasta_files = sorted(workspace.glob('human_fasta*.py'))
    spiral_files = sorted(workspace.glob('human_spiral*.py'))

    all_files = eco_files + fasta_files + spiral_files

    print("="*70)
    print("FIX COMMON VISUALIZATION ERRORS")
    print("="*70)
    print()
    print(f"Found {len(all_files)} scripts to check")
    print()

    fixed = 0

    for filepath in all_files:
        try:
            if process_file(filepath):
                fixed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Fixed: {fixed} files")
    print()

    if fixed > 0:
        print("✓ Common errors fixed!")
        print("  - Added 'N' (unknown nucleotide) to base_colors")
        print("  - Fixed canvas definition errors")

if __name__ == "__main__":
    main()
