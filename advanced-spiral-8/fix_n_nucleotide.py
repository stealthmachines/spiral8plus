"""Add 'N' nucleotide handling to scripts"""
from pathlib import Path

FILES_TO_FIX = [
    'human_eco19.py',
    'human_eco20.py',
    'human_eco21.py',
]

def fix_n_nucleotide(filepath):
    """Add 'N' to nucleotides list and handle it in mapping"""
    print(f"\nFixing: {filepath.name}")

    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content

        # Fix 1: Add 'N' to nucleotides list
        content = content.replace(
            'nucleotides = ["A", "T", "G", "C"]',
            'nucleotides = ["A", "T", "G", "C", "N"]'
        )

        # Fix 2: Handle 'N' in index lookup
        content = content.replace(
            'geom_idx = mapping[nucleotides.index(base)]',
            'geom_idx = 0 if base == "N" else mapping[nucleotides.index(base)]'
        )

        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"  [OK] Added 'N' nucleotide handling")
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
    print("BATCH FIX: Add 'N' nucleotide handling")
    print("=" * 70)

    for filename in FILES_TO_FIX:
        filepath = script_dir / filename
        if filepath.exists():
            if fix_n_nucleotide(filepath):
                fixed_count += 1
        else:
            print(f"\n[WARN] File not found: {filename}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(FILES_TO_FIX)} files")
    print("=" * 70)

if __name__ == '__main__':
    main()
