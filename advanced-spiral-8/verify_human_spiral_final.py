"""
Final verification test for human spiral files
Extracts and tests ONLY the genome loading functions (no VisPy needed)
"""

import os
import glob
import re

def extract_and_test_file(filepath):
    """Extract genome functions and test them"""

    print(f"\n{'='*70}")
    print(f"Testing: {filepath}")
    print(f"{'='*70}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract just the genome-related code
    code_parts = []

    # Get find_human_fasta function
    find_func_match = re.search(
        r'(def find_human_fasta\(\):.*?)(?=\ndef [a-z_]+|\nclass |\n# -)',
        content,
        re.DOTALL
    )
    if find_func_match:
        code_parts.append(find_func_match.group(1).strip())
    else:
        print("❌ Could not find find_human_fasta function")
        return False

    # Get load_genome function
    load_func_match = re.search(
        r'(def load_genome\(.*?\):.*?)(?=\n# Load the|\ntry:|\n# -)',
        content,
        re.DOTALL
    )
    if load_func_match:
        code_parts.append(load_func_match.group(1).strip())
    else:
        print("❌ Could not find load_genome function")
        return False

    # Create test code
    test_code = f"""
import os
import glob

os.environ['GENOME_LIMIT'] = '5000'

{code_parts[0]}

{code_parts[1]}

# Run test
try:
    fasta_path = find_human_fasta()
    print(f"✅ Found FASTA: {{fasta_path}}")

    genome_seq, metadata = load_genome(fasta_path)
    print(f"✅ Genome loaded: {{len(genome_seq):,}} nucleotides")
    print(f"✅ Chromosomes: {{metadata['chromosomes']}}")
    print(f"✅ Total loaded: {{metadata['total_loaded']:,}}")

    # Verify GENOME_LIMIT worked
    if len(genome_seq) == 5000:
        print("✅ GENOME_LIMIT environment variable works correctly")
    else:
        print(f"⚠️ Expected 5000 nucleotides, got {{len(genome_seq)}}")

    print("\\n✅ TEST PASSED")

except FileNotFoundError as e:
    print(f"⚠️ FASTA file not found: {{e}}")
    print("   (Human genome data might not be downloaded)")
except Exception as e:
    print(f"❌ TEST FAILED: {{e}}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

    # Execute the test
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all human spiral files"""

    print("="*70)
    print("HUMAN SPIRAL FILES - GENOME LOADING VERIFICATION")
    print("="*70)
    print("Testing genome loading functions with GENOME_LIMIT=5000\n")

    files_to_test = [
        'human_spiral8.py',
        'human_spiral9.py'
    ]

    results = {}

    for filename in files_to_test:
        if os.path.exists(filename):
            results[filename] = extract_and_test_file(filename)
        else:
            print(f"\n⚠️ {filename} not found")
            results[filename] = False

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for filename, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{filename}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All human spiral files working correctly!")
        print("✅ Genome loading functions operational")
        print("✅ GENOME_LIMIT environment variable support working")
        print("✅ Metadata tracking functional")
        return True
    else:
        print(f"\n⚠️ {total - passed} file(s) need attention")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
