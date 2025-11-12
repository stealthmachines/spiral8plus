"""Batch fix Unicode characters in Python scripts for Windows console compatibility"""
import re
from pathlib import Path

# Unicode replacements
REPLACEMENTS = [
    ('✅', '[OK]'),
    ('❌', '[ERROR]'),
    ('⚠️', '[WARN]'),
    ('⚠', '[WARN]'),
    ('🌀', '==='),
    ('🎵', '==='),
    ('🏆', '==='),
    ('🧬', '==='),
    ('🌈', '==='),
    ('🔬', '==='),
    ('→', '->'),
    ('φ', 'phi'),
    ('α', 'alpha'),
    ('Ω', 'Omega'),
    ('✓', '[OK]'),
]

# Files to fix
FILES_TO_FIX = [
    'human_waterfall_animation.py',
    'human_eco_unified_phi_synthesis.py',
    'human_spiral8.py',
    'human_spiral9.py',
]

def fix_file(filepath):
    """Replace Unicode characters in a file"""
    print(f"\nFixing: {filepath.name}")

    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content

        # Apply all replacements
        for old, new in REPLACEMENTS:
            if old in content:
                count = content.count(old)
                content = content.replace(old, new)
                print(f"  Replaced {count} instances of '{old}' -> '{new}'")

        # Write back if changes were made
        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"  [OK] File updated successfully")
            return True
        else:
            print(f"  No Unicode characters found")
            return False

    except Exception as e:
        print(f"  [ERROR] Failed to fix file: {e}")
        return False

def main():
    script_dir = Path(__file__).parent
    fixed_count = 0

    print("=" * 70)
    print("BATCH UNICODE FIX FOR WINDOWS CONSOLE COMPATIBILITY")
    print("=" * 70)

    for filename in FILES_TO_FIX:
        filepath = script_dir / filename
        if filepath.exists():
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"\n[WARN] File not found: {filename}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(FILES_TO_FIX)} files")
    print("=" * 70)

if __name__ == '__main__':
    main()
