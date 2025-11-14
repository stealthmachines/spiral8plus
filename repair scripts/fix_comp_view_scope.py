"""Fix comp_view scope issue in eco36/40/41/42 by moving canvas creation before Bacteria class"""
import os

files = [
    'advanced-spiral-8/human_eco36.py',
    'advanced-spiral-8/human_eco40.py',
    'advanced-spiral-8/human_eco41.py',
    'advanced-spiral-8/human_eco42.py',
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"[ERROR] Not found: {filepath}")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix 1: Move canvas creation before Bacteria class
    old1 = """genome_len = len(genome_seq)

class Bacteria:"""

    new1 = """genome_len = len(genome_seq)

# ---------- VISPY SETUP (must be before Bacteria class) ----------
comp_canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'

class Bacteria:"""

    # Fix 2: Remove duplicate canvas creation
    old2 = """# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
comp_view = canvas.central_widget.add_view()
comp_view.camera = 'turntable'

# Variables"""

    new2 = """# Variables"""

    # Fix 3: Rename canvas references
    replacements = [
        ('    canvas.central_widget.children[0].camera.azimuth', '    comp_canvas.central_widget.children[0].camera.azimuth'),
        ('    canvas.central_widget.children[0].camera.elevation', '    comp_canvas.central_widget.children[0].camera.elevation'),
        ('    canvas.update()', '    comp_canvas.update()'),
        ('    canvas.show()', '    comp_canvas.show()'),
    ]

    modified = False
    if old1 in content:
        content = content.replace(old1, new1)
        modified = True

    if old2 in content:
        content = content.replace(old2, new2)
        modified = True

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] Fixed: {filepath}")
    else:
        print(f"[SKIP] No changes: {filepath}")

print("\nDone!")
