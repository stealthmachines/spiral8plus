"""
Batch converter: Convert E. coli visualization files to SARS-CoV-2 versions
Creates COVID-19 versions of key visualization files with auto-detection
"""

import os
import re
import glob

# Auto-detect FASTA helper function to inject
AUTO_DETECT_CODE = '''
# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_covid_fasta():
    """Automatically find the COVID-19 FASTA file in the ncbi_dataset"""
    import glob
    possible_paths = [
        r"ncbi_dataset\\data\\GCF_009858895.2\\*.fna",
        r"ncbi_dataset\\data\\GCA_009858895.3\\*.fna",
        r"ncbi_dataset\\data\\*\\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find COVID-19 FASTA file in ncbi_dataset directory")
'''

# Files to convert (all files with ecoli_k12.fasta references)
FILES_TO_CONVERT = [
    'fasta16.py',
    'fasta18.py',
    'fasta19.py',
    'ecoli.py',
    'ecoli1.py',
    'ecoli2.py',
    'ecoli4.py',
    'ecoli10.py',
    'ecoli11.py',
    'ecoli12.py',
    'ecoli13.py',
    'ecoli14.py',
    'ecoli15.py',
    'ecoli16.py',
    'ecoli17.py',
    'ecoli18.py',
    'ecoli19.py',
    'ecoli20.py',
    'ecoli21.py',
    'ecoli22.py',
    'ecoli23.py',
    'ecoli24.py',
    'ecoli25.py',
    'ecoli26.py',
    'ecoli27.py',
    'ecoli28.py',
    'ecoli29.py',
    'ecoli30.py',
    'ecoli31.py',
    'ecoli32.py',
    'ecoli33.py',
    'ecoli34.py',
    'ecoli35.py',
    'ecoli36.py',
    'ecoli37.py',
    'ecoli38.py',
    'ecoli39.py',
    'ecoli40.py',
    'ecoli41.py',
    'ecoli42.py',
    'ecoli43.py',
    'ecoli44.py',
    'ecoli45.py',
    'ecoli46.py',
    'ecoli46_c_engine.py',
    'ecoli46_v2_100percent_fasta.py',
    'ecoli46_v3_ai_interpreter.py',
    'ecoli46_v3_gpu_full.py',
    'ecoli46_v3_pure_fasta.py',
    'ecoli46_v3_terminal.py',
    'ecoli47.py',
    'ecoli48.py',
    'ecoli_unified_phi_synthesis.py',
]

def convert_file(input_file, output_file):
    """Convert a single file from E. coli to COVID-19"""

    if not os.path.exists(input_file):
        print(f"  ⚠ Skipping {input_file} (not found)")
        return False

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already has auto-detect code
    if 'find_covid_fasta' in content:
        print(f"  ⚠ Skipping {input_file} (already converted)")
        return False

    # Add auto-detect function at the top (after imports)
    if 'def load_genome' in content:
        # Add before load_genome function
        content = content.replace('def load_genome', AUTO_DETECT_CODE + '\n\ndef load_genome')
    elif 'import' in content:
        # Add after the last import statement
        import_pattern = r'((?:^(?:from|import)\s+.*\n)+)'
        match = re.search(import_pattern, content, re.MULTILINE)
        if match:
            last_import_end = match.end()
            content = content[:last_import_end] + '\n' + AUTO_DETECT_CODE + '\n' + content[last_import_end:]

    # Replace genome loading calls BEFORE text replacement
    patterns_to_replace = [
        (r'genome_seq\s*=\s*load_genome\s*\(\s*["\']ecoli_k12\.fasta["\']\s*\)',
         'genome_seq = load_genome(find_covid_fasta())'),
        (r'fasta_path\s*=\s*["\']ecoli_k12\.fasta["\']',
         'fasta_path = find_covid_fasta()'),
        (r'fasta_file\s*=\s*["\']ecoli_k12\.fasta["\']',
         'fasta_file = find_covid_fasta()'),
        # SeqIO.read patterns
        (r'SeqIO\.read\s*\(\s*["\']ecoli_k12\.fasta["\']\s*,',
         'SeqIO.read(find_covid_fasta(),'),
        # Path object patterns
        (r'Path\s*\(__file__\)\s*\.\s*parent\s*/\s*["\']ecoli_k12\.fasta["\']',
         'find_covid_fasta()'),
        # Byte string patterns
        (r'fasta_path\s*=\s*b["\']ecoli_k12\.fasta["\']',
         'fasta_path = find_covid_fasta().encode()'),
    ]

    for pattern, replacement in patterns_to_replace:
        content = re.sub(pattern, replacement, content)

    # Replace title/comments (after genome path replacement)
    content = re.sub(r'\bE\. coli\b|\bE\.coli\b', 'SARS-CoV-2', content)
    # Don't replace 'ecoli' in variable names or comments that are code-related    # Update background colors for better viral visualization
    content = content.replace("bgcolor='black'", "bgcolor='#000011'")
    content = content.replace('bgcolor="black"', 'bgcolor="#000011"')

    # Add genome info print if main block exists
    if "if __name__ == '__main__':" in content and 'print' in content:
        # Try to add helpful output
        content = content.replace(
            "if __name__ == '__main__':",
            '''if __name__ == '__main__':
    print("\\n" + "="*60)
    print("SARS-CoV-2 Wuhan-Hu-1 Genome Visualization")
    print("="*60)
    if 'genome_len' in dir():
        print(f"Genome length: {genome_len:,} nucleotides")
    print("="*60)
'''
        )

    # Write converted file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return True

def main():
    """Main conversion process"""
    print("\n" + "="*70)
    print("BATCH CONVERSION: E. coli → SARS-CoV-2 Visualization Files")
    print("="*70 + "\n")

    converted = []
    skipped = []

    for filename in FILES_TO_CONVERT:
        input_path = filename
        # Create covid_ version
        base = filename.replace('.py', '')
        output_path = f'covid_{base}.py'

        print(f"Converting: {filename} → {output_path}")

        if convert_file(input_path, output_path):
            converted.append(output_path)
            print(f"  ✓ Success\n")
        else:
            skipped.append(filename)
            print()

    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"✓ Converted: {len(converted)} files")
    print(f"⚠ Skipped: {len(skipped)} files")

    if converted:
        print("\nConverted files:")
        for f in converted:
            print(f"  • {f}")

    if skipped:
        print("\nSkipped files:")
        for f in skipped:
            print(f"  • {f}")

    print("\n" + "="*70)
    print("Next step: Run batch_test_covid.py to verify all files")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
