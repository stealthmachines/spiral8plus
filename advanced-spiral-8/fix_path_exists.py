"""Fix Path.exists() -> os.path.exists() in v3 files"""
from pathlib import Path

FILES_TO_FIX = [
    'human_eco46_v3_gpu_full.py',
    'human_eco46_v3_pure_fasta.py',
    'human_eco46_v3_terminal.py',
]

def fix_path_exists(filepath):
    """Replace fasta_path.exists() with os.path.exists(fasta_path) and add os import"""
    print(f"\nFixing: {filepath.name}")

    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content

        # Fix 1: Add os import if not present
        if 'import os' not in content:
            # Find the import section and add os
            if 'import sys' in content:
                content = content.replace('import sys\n', 'import sys\nimport os\n')
                print(f"  Added: import os")

        # Fix 2: Replace .exists() with os.path.exists()
        if 'fasta_path.exists()' in content:
            content = content.replace('fasta_path.exists()', 'os.path.exists(fasta_path)')
            print(f"  Fixed: fasta_path.exists() -> os.path.exists(fasta_path)")

        if 'dll_path.exists()' in content:
            content = content.replace('dll_path.exists()', 'os.path.exists(str(dll_path))')
            print(f"  Fixed: dll_path.exists() -> os.path.exists(str(dll_path))")

        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"  [OK] File updated successfully")
            return True
        else:
            print(f"  [WARN] No changes needed")
            return False

    except Exception as e:
        print(f"  [ERROR] Failed to fix file: {e}")
        return False

def main():
    script_dir = Path(__file__).parent
    fixed_count = 0

    print("=" * 70)
    print("BATCH FIX: Path.exists() -> os.path.exists()")
    print("=" * 70)

    for filename in FILES_TO_FIX:
        filepath = script_dir / filename
        if filepath.exists():
            if fix_path_exists(filepath):
                fixed_count += 1
        else:
            print(f"\n[WARN] File not found: {filename}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(FILES_TO_FIX)} files")
    print("=" * 70)

if __name__ == '__main__':
    main()
