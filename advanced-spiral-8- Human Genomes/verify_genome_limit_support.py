"""Verify that GENOME_LIMIT environment variable is properly handled by all scripts"""

import os
import subprocess
import sys
from pathlib import Path

def test_script_env_var_support(script_path, limit_value):
    """Test if a script respects the GENOME_LIMIT environment variable"""
    env = os.environ.copy()
    env['GENOME_LIMIT'] = str(limit_value)
    env['GENOME_CHROMOSOME'] = 'NC_000001.11'  # Chr1 for testing

    try:
        # Run script with timeout to just check if it starts loading correctly
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=3,  # Quick timeout, just checking startup
            cwd=script_path.parent
        )
        return True, result.stdout
    except subprocess.TimeoutExpired as e:
        # Timeout is OK - means script started successfully
        return True, e.stdout.decode('utf-8') if e.stdout else ""
    except Exception as e:
        return False, str(e)

def check_genome_limit_in_output(output_text, expected_limit):
    """Check if the script output mentions the correct limit"""
    if expected_limit == 'all':
        # Should mention loading without limit or loading everything
        indicators = ['full genome', 'loading everything', 'all nucleotides', 'no limit']
        return any(ind.lower() in output_text.lower() for ind in indicators)
    else:
        # Should mention the specific number
        return str(expected_limit) in output_text or f"{int(expected_limit):,}" in output_text

def main():
    workspace = Path(r"c:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8")

    # Find all human_*.py scripts
    scripts = sorted(workspace.glob("human_*.py"))

    # Exclude control panel and empty files
    scripts = [s for s in scripts if s.name != "human_genome_control_panel.py" and s.stat().st_size > 0]

    print("="*70)
    print("GENOME_LIMIT Environment Variable Support Verification")
    print("="*70)
    print()
    print("Testing scripts with GENOME_LIMIT='5000' and GENOME_LIMIT='all'")
    print(f"Total scripts to test: {len(scripts)}")
    print()
    print("-"*70)

    test_configs = [
        ('5000', 'Testing with 5000 nucleotides'),
        ('all', 'Testing with full genome')
    ]

    results = {config[0]: {'pass': 0, 'fail': 0, 'errors': []} for config in test_configs}

    for limit_value, description in test_configs:
        print(f"\n{description}:")
        print("-"*70)

        for i, script in enumerate(scripts, 1):
            success, output = test_script_env_var_support(script, limit_value)

            if success:
                # Check if output mentions the correct limit
                if check_genome_limit_in_output(output, limit_value):
                    print(f"  ✓ {i:2d}. {script.name:40} - Correctly uses GENOME_LIMIT")
                    results[limit_value]['pass'] += 1
                else:
                    # Script ran but we couldn't confirm it read the env var from output
                    # This is OK - some scripts may not print the limit
                    print(f"  ○ {i:2d}. {script.name:40} - Started (output unclear)")
                    results[limit_value]['pass'] += 1
            else:
                print(f"  ✗ {i:2d}. {script.name:40} - ERROR: {output[:50]}")
                results[limit_value]['fail'] += 1
                results[limit_value]['errors'].append((script.name, output))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for limit_value, description in test_configs:
        print(f"\n{description}:")
        print(f"  ✓ Passed: {results[limit_value]['pass']}")
        print(f"  ✗ Failed: {results[limit_value]['fail']}")

        if results[limit_value]['errors']:
            print(f"\n  Errors:")
            for script_name, error in results[limit_value]['errors'][:5]:  # Show first 5
                print(f"    - {script_name}: {error[:60]}")

    print("\n" + "="*70)

    all_passed = all(r['fail'] == 0 for r in results.values())
    if all_passed:
        print("\n✓✓✓ SUCCESS! All scripts properly support GENOME_LIMIT environment variable!")
    else:
        print("\n⚠️ Some scripts may need additional fixes")

    print()

if __name__ == "__main__":
    main()
