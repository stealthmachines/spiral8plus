"""
Quick syntax test for all human_fasta*.py files
"""

import os
import sys
from pathlib import Path

def test_syntax(filepath):
    """Test if a file has syntax errors"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def main():
    """Test all human_fasta files"""

    script_dir = Path(__file__).parent

    # Find all human_fasta*.py files
    fasta_files = sorted(script_dir.glob('human_fasta*.py'))

    print("="*70)
    print(f"SYNTAX TEST: Human Fasta Files ({len(fasta_files)} files)")
    print("="*70)

    passed = 0
    failed = []

    for filepath in fasta_files:
        ok, error = test_syntax(filepath)

        if ok:
            print(f"✅ {filepath.name}")
            passed += 1
        else:
            print(f"❌ {filepath.name}: {error}")
            failed.append((filepath.name, error))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(fasta_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\n❌ FILES WITH ERRORS:")
        for filename, error in failed:
            print(f"  • {filename}: {error}")
        return False
    else:
        print(f"\n🎉 All files passed syntax check!")
        return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
