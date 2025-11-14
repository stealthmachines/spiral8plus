"""Fix 'N' nucleotide handling - map to 'A' as default"""
from pathlib import Path

FILES_TO_FIX = [
    'human_eco16.py',
    'human_eco19.py',
    'human_eco20.py',
    'human_eco21.py',
]

def fix_n_nucleotide_proper(filepath):
    """Remove 'N' from nucleotides list and map it to 'A' instead"""
    print(f"\nFixing: {filepath.name}")

    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content

        # Fix 1: Revert nucleotides list back to 4 bases only
        content = content.replace(
            'nucleotides = ["A", "T", "G", "C", "N"]',
            'nucleotides = ["A", "T", "G", "C"]'
        )

        # Fix 2: Map 'N' to 'A' before lookup
        content = content.replace(
            'geom_idx = 0 if base == "N" else mapping[nucleotides.index(base)]',
            'base_normalized = "A" if base == "N" else base\n        geom_idx = mapping[nucleotides.index(base_normalized)]'
        )

        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"  [OK] Fixed 'N' to map to 'A' (default base)")
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
    print("BATCH FIX: Proper 'N' nucleotide handling (map to 'A')")
    print("=" * 70)

    for filename in FILES_TO_FIX:
        filepath = script_dir / filename
        if filepath.exists():
            if fix_n_nucleotide_proper(filepath):
                fixed_count += 1
        else:
            print(f"\n[WARN] File not found: {filename}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(FILES_TO_FIX)} files")
    print("=" * 70)

if __name__ == '__main__':
    main()
