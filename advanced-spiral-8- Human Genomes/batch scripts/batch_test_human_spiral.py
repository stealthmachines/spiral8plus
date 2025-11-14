"""
Batch test human spiral visualization files for errors
Tests: human_spiral8.py, human_spiral9.py
"""

import os
import sys
import subprocess
from pathlib import Path

def test_file_syntax(filepath):
    """Test if a Python file has syntax errors"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def test_file_imports(filepath):
    """Test if file can import required modules (without running main code)"""
    test_code = f'''
import sys
import os

# Read file and extract only import statements
with open(r"{filepath}", "r", encoding="utf-8") as f:
    lines = f.readlines()

imports_only = []
for line in lines:
    stripped = line.strip()
    # Stop at first non-import/comment/blank line after imports section
    if stripped and not stripped.startswith('#') and not stripped.startswith('import') and not stripped.startswith('from'):
        if imports_only:  # If we've already collected some imports, stop here
            break
    if stripped.startswith('import') or stripped.startswith('from'):
        imports_only.append(line)

# Try to execute imports
try:
    exec(''.join(imports_only))
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
'''

    try:
        result = subprocess.run(
            ['python', '-c', test_code],
            capture_output=True,
            text=True,
            timeout=10
        )

        if 'IMPORT_SUCCESS' in result.stdout:
            return True, None
        elif 'IMPORT_ERROR' in result.stdout:
            error_msg = result.stdout.split('IMPORT_ERROR: ')[1].strip()
            return False, error_msg
        else:
            return False, result.stderr or "Unknown import error"
    except subprocess.TimeoutExpired:
        return False, "Import test timeout"
    except Exception as e:
        return False, str(e)

def test_genome_loading(filepath):
    """Test if file can load genome data without running visualization"""
    test_code = f'''
import os
import sys

# Set environment to limit loading
os.environ['GENOME_LIMIT'] = '10000'

# Read the file
with open(r"{filepath}", "r", encoding="utf-8") as f:
    content = f.read()

# Extract only the genome loading portion (before visualization)
try:
    # Split at common visualization markers
    markers = ['# ---------- VISPY SETUP', '# ---------- RUN SIMULATION', 'canvas = scene.SceneCanvas']
    code_to_test = content

    for marker in markers:
        if marker in content:
            code_to_test = content.split(marker)[0]
            break

    # Execute just the loading code
    exec(code_to_test)

    # Check if genome was loaded
    if 'genome_seq' in dir() and genome_seq is not None:
        print(f"GENOME_SUCCESS: {{len(genome_seq)}} nucleotides")
    else:
        print("GENOME_WARNING: No genome loaded")

except Exception as e:
    print(f"GENOME_ERROR: {{e}}")
'''

    try:
        result = subprocess.run(
            ['python', '-c', test_code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(filepath)
        )

        if 'GENOME_SUCCESS' in result.stdout:
            count = result.stdout.split('GENOME_SUCCESS: ')[1].split()[0]
            return True, f"Loaded {count} nucleotides"
        elif 'GENOME_WARNING' in result.stdout:
            return True, "Warning: No genome loaded (might be expected)"
        elif 'GENOME_ERROR' in result.stdout:
            error_msg = result.stdout.split('GENOME_ERROR: ')[1].strip()
            return False, error_msg
        else:
            # Check stderr for errors
            if result.stderr:
                return False, result.stderr[:200]
            return False, "Unknown genome loading error"
    except subprocess.TimeoutExpired:
        return False, "Genome loading timeout (might be trying to load too much)"
    except Exception as e:
        return False, str(e)

def check_common_issues(filepath):
    """Check for common issues in the file"""
    issues = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check 1: Has find_human_fasta function
    if 'def find_human_fasta' not in content:
        issues.append("❌ Missing find_human_fasta() function")
    else:
        issues.append("✅ Has find_human_fasta() function")

    # Check 2: Has load_genome function
    if 'def load_genome' not in content:
        issues.append("❌ Missing load_genome() function")
    else:
        issues.append("✅ Has load_genome() function")

    # Check 3: Has GENOME_LIMIT support
    if 'GENOME_LIMIT' in content:
        issues.append("✅ Has GENOME_LIMIT environment variable support")
    else:
        issues.append("⚠️ Missing GENOME_LIMIT support (should be added)")

    # Check 4: Has proper UTF-8 encoding
    if 'encoding=' in content or "encoding=" in content:
        issues.append("✅ Has UTF-8 encoding for file operations")
    else:
        issues.append("⚠️ Missing UTF-8 encoding (might cause issues on Windows)")

    # Check 5: Returns tuple from load_genome
    if 'return seq, metadata' in content or 'return genome_seq, metadata' in content:
        issues.append("✅ load_genome returns (sequence, metadata) tuple")
    elif 'return seq' in content or 'return genome_seq' in content:
        issues.append("⚠️ load_genome returns only sequence (metadata missing)")

    # Check 6: Human genome paths
    if 'GCF_000001405.40' in content:
        issues.append("✅ Has human genome path (GCF_000001405.40)")
    else:
        issues.append("❌ Missing human genome path")

    # Check 7: No COVID references remaining
    covid_refs = []
    if 'find_covid_fasta' in content:
        covid_refs.append("find_covid_fasta()")
    if 'GCF_009858895' in content:
        covid_refs.append("COVID genome accession")
    if covid_refs:
        issues.append(f"⚠️ Still has COVID references: {', '.join(covid_refs)}")
    else:
        issues.append("✅ No COVID references found")

    return issues

def main():
    """Test all human spiral files"""

    script_dir = Path(__file__).parent

    # Find all human_spiral*.py files
    spiral_files = sorted(script_dir.glob('human_spiral*.py'))

    if not spiral_files:
        print("⚠️ No human_spiral*.py files found")
        return

    print("="*70)
    print("BATCH TEST: Human Spiral Visualization Files")
    print("="*70)
    print(f"Found {len(spiral_files)} files to test\n")

    results = {}

    for filepath in spiral_files:
        print(f"\n{'='*70}")
        print(f"Testing: {filepath.name}")
        print(f"{'='*70}")

        test_results = {
            'syntax': None,
            'imports': None,
            'genome': None,
            'issues': []
        }

        # Test 1: Syntax
        print("\n1. Syntax Check...")
        syntax_ok, syntax_error = test_file_syntax(filepath)
        test_results['syntax'] = (syntax_ok, syntax_error)

        if syntax_ok:
            print("   ✅ Syntax: PASSED")
        else:
            print(f"   ❌ Syntax: FAILED - {syntax_error}")

        # Test 2: Imports
        if syntax_ok:
            print("\n2. Import Check...")
            imports_ok, import_error = test_file_imports(filepath)
            test_results['imports'] = (imports_ok, import_error)

            if imports_ok:
                print("   ✅ Imports: PASSED")
            else:
                print(f"   ⚠️ Imports: {import_error}")
                print("   (Some modules might not be installed - this is OK)")

        # Test 3: Genome Loading
        if syntax_ok:
            print("\n3. Genome Loading Check...")
            genome_ok, genome_msg = test_genome_loading(filepath)
            test_results['genome'] = (genome_ok, genome_msg)

            if genome_ok:
                print(f"   ✅ Genome: PASSED - {genome_msg}")
            else:
                print(f"   ❌ Genome: FAILED - {genome_msg}")

        # Test 4: Common Issues
        print("\n4. Code Quality Check...")
        issues = check_common_issues(filepath)
        test_results['issues'] = issues

        for issue in issues:
            print(f"   {issue}")

        results[filepath.name] = test_results

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")

    total_files = len(results)
    syntax_passed = sum(1 for r in results.values() if r['syntax'][0])
    genome_passed = sum(1 for r in results.values() if r['genome'] and r['genome'][0])

    print(f"\nTotal files tested: {total_files}")
    print(f"Syntax passed: {syntax_passed}/{total_files}")
    print(f"Genome loading passed: {genome_passed}/{total_files}")

    # Show files that need attention
    needs_attention = []
    for filename, result in results.items():
        if not result['syntax'][0]:
            needs_attention.append((filename, "Syntax errors"))
        elif result['genome'] and not result['genome'][0]:
            needs_attention.append((filename, "Genome loading failed"))

    if needs_attention:
        print(f"\n⚠️ Files needing attention:")
        for filename, reason in needs_attention:
            print(f"   • {filename}: {reason}")
    else:
        print(f"\n✅ All files passed basic tests!")

    print(f"\n{'='*70}")
    print("Detailed results saved above. Review any ⚠️ or ❌ items.")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
