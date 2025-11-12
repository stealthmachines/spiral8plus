"""Initialize collections dict before update() function"""
from pathlib import Path

FILES_TO_FIX = [
    'human_eco25.py',
    'human_eco28.py',
    'human_eco32.py',
    'human_eco33.py',
]

def add_collections_init(filepath):
    """Add collections dictionary initialization"""
    print(f"\nFixing: {filepath.name}")

    try:
        content = filepath.read_text(encoding='utf-8')

        # Find the line with "rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []"
        if 'rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []' in content:
            # Insert collections dict initialization before that line
            old_section = 'rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []'
            new_section = '''# Collections dictionary
collections = {
    'rungs': [],
    'echoes': [],
    'links': [],
    'labels': [],
    'centers': [],
    'emerged': []
}

rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []'''

            content = content.replace(old_section, new_section)
            filepath.write_text(content, encoding='utf-8')
            print(f"  [OK] Added collections dictionary initialization")
            return True
        else:
            print(f"  [WARN] Pattern not found")
            return False

    except Exception as e:
        print(f"  [ERROR] Failed to fix file: {e}")
        return False

def main():
    script_dir = Path(__file__).parent
    fixed_count = 0

    print("=" * 70)
    print("BATCH FIX: Add collections dictionary initialization")
    print("=" * 70)

    for filename in FILES_TO_FIX:
        filepath = script_dir / filename
        if filepath.exists():
            if add_collections_init(filepath):
                fixed_count += 1
        else:
            print(f"\n[WARN] File not found: {filename}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(FILES_TO_FIX)} files")
    print("=" * 70)

if __name__ == '__main__':
    main()
