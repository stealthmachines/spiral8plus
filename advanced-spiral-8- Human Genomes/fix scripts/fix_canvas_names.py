"""Batch fix NameError issues - rename canvas/view to comp_canvas/comp_view"""
from pathlib import Path
import re

FILES_TO_FIX = [
    'human_eco30.py',
    'human_eco35.py',
    'human_eco36.py',
    'human_eco40.py',
    'human_eco41.py',
    'human_eco42.py',
]

def fix_canvas_names(filepath):
    """Rename canvas -> comp_canvas and view -> comp_view in setup section"""
    print(f"\nFixing: {filepath.name}")

    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content

        # Find the VISPY SETUP section
        if '# ---------- VISPY SETUP ----------' in content:
            lines = content.split('\n')
            modified = False

            for i, line in enumerate(lines):
                # Only modify in VISPY SETUP section (look for early initialization)
                if i < 200:  # Within first 200 lines
                    # Change canvas = to comp_canvas =
                    if 'canvas = scene.SceneCanvas' in line:
                        lines[i] = line.replace('canvas = scene.SceneCanvas', 'comp_canvas = scene.SceneCanvas')
                        print(f"  Line {i+1}: canvas -> comp_canvas")
                        modified = True

                    # Change view = canvas. to comp_view = comp_canvas.
                    elif 'view = canvas.central_widget' in line:
                        lines[i] = line.replace('view = canvas.central_widget', 'comp_view = comp_canvas.central_widget')
                        print(f"  Line {i+1}: view -> comp_view")
                        modified = True

                    # Change view.camera = to comp_view.camera =
                    elif line.strip().startswith('view.camera = '):
                        lines[i] = line.replace('view.camera = ', 'comp_view.camera = ')
                        print(f"  Line {i+1}: view.camera -> comp_view.camera")
                        modified = True

                    # Change parent=view.scene to parent=comp_view.scene in Line() calls
                    elif 'parent=view.scene' in line and ('Line(' in line or 'strand' in line):
                        lines[i] = line.replace('parent=view.scene', 'parent=comp_view.scene')
                        print(f"  Line {i+1}: parent=view.scene -> parent=comp_view.scene")
                        modified = True

            if modified:
                content = '\n'.join(lines)
                filepath.write_text(content, encoding='utf-8')
                print(f"  [OK] File updated successfully")
                return True
            else:
                print(f"  No changes needed")
                return False
        else:
            print(f"  [WARN] No VISPY SETUP section found")
            return False

    except Exception as e:
        print(f"  [ERROR] Failed to fix file: {e}")
        return False

def main():
    script_dir = Path(__file__).parent
    fixed_count = 0

    print("=" * 70)
    print("BATCH FIX: canvas/view -> comp_canvas/comp_view")
    print("=" * 70)

    for filename in FILES_TO_FIX:
        filepath = script_dir / filename
        if filepath.exists():
            if fix_canvas_names(filepath):
                fixed_count += 1
        else:
            print(f"\n[WARN] File not found: {filename}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(FILES_TO_FIX)} files")
    print("=" * 70)

if __name__ == '__main__':
    main()
