"""Fix comp_view/comp_canvas issues in eco40/41/42"""
import os

files = [
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

    # Fix 1: Add canvas/view creation after genome loading
    old1 = """# ---------- MULTI-E.co ENVIRONMENT ----------
num_bacteria = 3
offsets = [np.array([i*30.0,0,0]) for i in range(num_bacteria)]
bacteria = [EColi(genome_seq, offset=o) for o in offsets]


# Variables"""

    new1 = """# ---------- VISPY SETUP (must be before EColi class) ----------
comp_canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'

# ---------- MULTI-E.co ENVIRONMENT ----------
num_bacteria = 3
offsets = [np.array([i*30.0,0,0]) for i in range(num_bacteria)]
bacteria = [EColi(genome_seq, offset=o) for o in offsets]


# Variables"""

    # Fix 2: Change view to comp_view in update function
    old2_list = [
        ('    view.camera.azimuth', '    comp_view.camera.azimuth'),
        ('    view.camera.elevation', '    comp_view.camera.elevation'),
        ('    canvas.update()', '    comp_canvas.update()'),
        ('    canvas.show()', '    comp_canvas.show()'),
    ]

    modified = False
    if old1 in content:
        content = content.replace(old1, new1)
        modified = True
        print(f"[OK] Added canvas/view: {filepath}")

    for old, new in old2_list:
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
