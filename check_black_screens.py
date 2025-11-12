"""Check for common black screen issues in scripts"""
import os
import re

scripts_to_check = [
    'advanced-spiral-8/human_eco29.py',
    'advanced-spiral-8/human_eco32.py',
    'advanced-spiral-8/human_eco33.py',
    'advanced-spiral-8/human_eco34.py',
    'advanced-spiral-8/human_fasta3.py',
    'advanced-spiral-8/human_fasta4.py',
    'advanced-spiral-8/human_fasta5.py',
    'advanced-spiral-8/human_fasta6.py',
    'advanced-spiral-8/human_fasta7.py',
]

print("Checking for common issues...\n")

for script in scripts_to_check:
    if not os.path.exists(script):
        print(f"[SKIP] {script} - not found")
        continue

    with open(script, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []

    # Check for undefined variables used in code
    undefined_vars = []
    if 'membrane_radius' in content:
        if re.search(r'\bmembrane_radius\s*=', content) is None:
            undefined_vars.append('membrane_radius')

    if 'percent_text' in content:
        if re.search(r'\bpercent_text\s*=', content) is None:
            undefined_vars.append('percent_text')

    # Check for canvas creation
    has_canvas = 'SceneCanvas' in content or 'canvas =' in content
    has_show = 'canvas.show()' in content or 'comp_canvas.show()' in content

    # Check for visual objects
    has_line = '= Line(' in content
    has_markers = '= Markers(' in content
    has_visuals = has_line or has_markers

    if undefined_vars:
        issues.append(f"Undefined vars: {', '.join(undefined_vars)}")

    if not has_canvas:
        issues.append("No canvas creation")

    if not has_show:
        issues.append("No canvas.show()")

    if not has_visuals:
        issues.append("No visual objects (Line/Markers)")

    if issues:
        print(f"[ISSUES] {os.path.basename(script)}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"[OK] {os.path.basename(script)}")

print("\nDone!")
