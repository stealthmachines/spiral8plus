"""Fix remaining Unicode emoji characters in 4 scripts"""
import os
import re

def fix_file(filepath, replacements):
    """Apply replacements to a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    for old, new in replacements:
        content = content.replace(old, new)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# Unicode replacements
replacements = [
    ('✅', '[OK]'),
    ('❌', '[ERROR]'),
    ('⚠️', '[WARN]'),
    ('🏆', '[SUCCESS]'),
    ('φ', 'phi'),
]

files = [
    'advanced-spiral-8/human_cross_cavity_tuning.py',
    'advanced-spiral-8/human_eco19.py',
    'advanced-spiral-8/human_eco20.py',
    'advanced-spiral-8/human_eco21.py',
]

print("Fixing remaining Unicode errors...")
for file in files:
    if os.path.exists(file):
        if fix_file(file, replacements):
            print(f"[OK] Fixed: {file}")
        else:
            print(f"[SKIP] No changes: {file}")
    else:
        print(f"[ERROR] Not found: {file}")

print("\nDone!")
