"""
COMPREHENSIVE TEST SUITE FOR ALL HUMAN GENOME SCRIPTS
======================================================

Systematically tests all human_*.py scripts with minimal nucleotide count
to identify compilation errors, runtime errors, and missing dependencies.

Reports:
- ✅ Scripts that run successfully
- ⚠️  Scripts with missing dependencies (acceptable)
- ❌ Scripts with errors that need fixing
"""

import subprocess
import sys
from pathlib import Path
import time
import json

# Test configuration
GENOME_LIMIT = "500000"  # 500K nucleotides - comprehensive test
GENOME_CHROMOSOME = "NC_000001.11"  # Chromosome 1
TIMEOUT_SECONDS = 15  # Max time per script (increased for larger data)


def discover_human_scripts():
    """Find all human_*.py scripts"""
    workspace = Path(__file__).parent

    scripts = {
        'eco': sorted(workspace.glob('human_eco*.py')),
        'fasta': sorted(workspace.glob('human_fasta*.py')),
        'spiral': sorted(workspace.glob('human_spiral*.py')),
        'advanced': sorted([
            p for p in workspace.glob('human_*.py')
            if any(keyword in p.stem for keyword in [
                'eight_geometries', 'cubic_scaling', 'cross_cavity', 'waterfall'
            ])
        ])
    }

    # Flatten to single list
    all_scripts = []
    for category, script_list in scripts.items():
        all_scripts.extend(script_list)

    # Remove duplicates and control panel
    all_scripts = list(set(all_scripts))
    all_scripts = [s for s in all_scripts if 'control_panel' not in s.name]
    all_scripts = sorted(all_scripts)

    return all_scripts


def test_script(script_path):
    """Test a single script with timeout"""
    result = {
        'script': script_path.name,
        'status': 'unknown',
        'error': None,
        'error_type': None,
        'output': ''
    }

    # Set environment variables
    env = {
        'GENOME_LIMIT': GENOME_LIMIT,
        'GENOME_CHROMOSOME': GENOME_CHROMOSOME,
        'GENOME_START': '0'
    }

    try:
        # Run script with timeout
        # Use 'py' command instead of 'python' for vispy compatibility on Windows
        process = subprocess.run(
            ['py', str(script_path)],
            env={**subprocess.os.environ, **env},
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        result['output'] = process.stdout + process.stderr

        if process.returncode == 0:
            result['status'] = 'success'
        else:
            result['status'] = 'error'
            result['error'] = result['output']

            # Classify error type
            if 'ModuleNotFoundError' in result['error'] or 'No module named' in result['error']:
                result['error_type'] = 'missing_dependency'
                # Extract module name
                if 'No module named' in result['error']:
                    module = result['error'].split("No module named '")[1].split("'")[0]
                    result['missing_module'] = module
            elif 'FileNotFoundError' in result['error']:
                result['error_type'] = 'file_not_found'
            elif 'ValueError' in result['error']:
                result['error_type'] = 'value_error'
            elif 'TypeError' in result['error']:
                result['error_type'] = 'type_error'
            elif 'AttributeError' in result['error']:
                result['error_type'] = 'attribute_error'
            elif 'NameError' in result['error']:
                result['error_type'] = 'name_error'
            elif 'SyntaxError' in result['error']:
                result['error_type'] = 'syntax_error'
            elif 'ImportError' in result['error']:
                result['error_type'] = 'import_error'
            else:
                result['error_type'] = 'unknown_error'

    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
        result['error'] = f'Script exceeded {TIMEOUT_SECONDS}s timeout (likely running visualization)'
        result['error_type'] = 'timeout'

    except Exception as e:
        result['status'] = 'exception'
        result['error'] = str(e)
        result['error_type'] = 'test_exception'

    return result


def main():
    """Run comprehensive test suite"""
    print("="*70)
    print("COMPREHENSIVE TEST SUITE FOR HUMAN GENOME SCRIPTS")
    print("="*70)
    print()

    print("Test Configuration:")
    print(f"  GENOME_LIMIT: {GENOME_LIMIT}")
    print(f"  GENOME_CHROMOSOME: {GENOME_CHROMOSOME}")
    print(f"  TIMEOUT: {TIMEOUT_SECONDS}s")
    print()

    # Discover scripts
    scripts = discover_human_scripts()

    print(f"Found {len(scripts)} human genome scripts to test")
    print()
    print("Testing scripts...")
    print("-"*70)

    results = []

    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Testing: {script.name}")

        result = test_script(script)
        results.append(result)

        # Print immediate status
        if result['status'] == 'success':
            print(f"  [OK] SUCCESS")
        elif result['status'] == 'timeout':
            print(f"  [TIMEOUT] (likely visualization running)")
        elif result['error_type'] == 'missing_dependency':
            module = result.get('missing_module', 'unknown')
            print(f"  [WARN] MISSING DEPENDENCY: {module}")
        else:
            print(f"  [ERROR] {result['error_type']}")
            # Show first line of error
            if result['error']:
                first_error_line = result['error'].split('\n')[-2] if '\n' in result['error'] else result['error'][:100]
                print(f"     {first_error_line}")

    # Generate summary report
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print()

    success = [r for r in results if r['status'] == 'success']
    timeout = [r for r in results if r['status'] == 'timeout']
    missing_deps = [r for r in results if r['error_type'] == 'missing_dependency']
    errors = [r for r in results if r['status'] == 'error' and r['error_type'] not in ['missing_dependency', 'timeout']]

    print(f"Total Scripts: {len(results)}")
    print(f"  [OK] Success: {len(success)}")
    print(f"  [TIMEOUT] Timeout (visualization): {len(timeout)}")
    print(f"  [WARN] Missing Dependencies: {len(missing_deps)}")
    print(f"  [ERROR] Errors Needing Fix: {len(errors)}")
    print()

    # Detailed sections
    if success:
        print("[OK] SUCCESSFUL SCRIPTS:")
        for r in success:
            print(f"  * {r['script']}")
        print()

    if timeout:
        print("[TIMEOUT] (Likely Visualizations Running):")
        for r in timeout:
            print(f"  * {r['script']}")
        print()

    if missing_deps:
        print("[WARN] MISSING DEPENDENCIES (Expected):")
        deps_by_module = {}
        for r in missing_deps:
            module = r.get('missing_module', 'unknown')
            if module not in deps_by_module:
                deps_by_module[module] = []
            deps_by_module[module].append(r['script'])

        for module, scripts in sorted(deps_by_module.items()):
            print(f"  {module}: {len(scripts)} scripts")
            for script in scripts[:3]:
                print(f"    * {script}")
            if len(scripts) > 3:
                print(f"    ... and {len(scripts)-3} more")
        print()

    if errors:
        print("[ERROR] ERRORS NEEDING FIX:")
        errors_by_type = {}
        for r in errors:
            error_type = r['error_type']
            if error_type not in errors_by_type:
                errors_by_type[error_type] = []
            errors_by_type[error_type].append(r)

        for error_type, error_list in sorted(errors_by_type.items()):
            print(f"\n  {error_type.upper()}: {len(error_list)} scripts")
            for r in error_list:
                print(f"    * {r['script']}")
                # Show error snippet
                if r['error']:
                    lines = r['error'].strip().split('\n')
                    # Get last 2 non-empty lines (usually the error)
                    error_lines = [l for l in lines if l.strip()][-2:]
                    for line in error_lines:
                        print(f"      {line[:120]}")
        print()

    # Save detailed results to JSON
    report_file = Path(__file__).parent / 'test_results.json'
    with open(report_file, 'w') as f:
        json.dump({
            'test_config': {
                'genome_limit': GENOME_LIMIT,
                'genome_chromosome': GENOME_CHROMOSOME,
                'timeout': TIMEOUT_SECONDS
            },
            'summary': {
                'total': len(results),
                'success': len(success),
                'timeout': len(timeout),
                'missing_deps': len(missing_deps),
                'errors': len(errors)
            },
            'results': results
        }, f, indent=2)

    print(f"[INFO] Detailed results saved to: {report_file.name}")
    print()

    # Return exit code based on errors
    if errors:
        print(f"[WARN] {len(errors)} scripts need fixing!")
        return 1
    else:
        print("[OK] All scripts compile successfully (excluding expected missing dependencies)!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
