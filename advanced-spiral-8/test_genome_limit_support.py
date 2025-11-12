import os

# Set environment variables
os.environ['GENOME_LIMIT'] = '10000'
os.environ['GENOME_CHROMOSOME'] = 'NC_000001.11'

print("Testing human_eco10.py GENOME_LIMIT support...")
print(f"GENOME_LIMIT: {os.environ.get('GENOME_LIMIT')}")
print(f"GENOME_CHROMOSOME: {os.environ.get('GENOME_CHROMOSOME')}")
print()

# Check if the file has GENOME_LIMIT support
with open('human_eco10.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

    if 'GENOME_LIMIT' in content:
        print("✅ human_eco10.py HAS GENOME_LIMIT support!")

        # Count occurrences
        count = content.count('GENOME_LIMIT')
        print(f"   Found {count} references to GENOME_LIMIT")

        if 'GENOME_CHROMOSOME' in content:
            print("✅ human_eco10.py HAS GENOME_CHROMOSOME support!")

        if 'GENOME_START' in content:
            print("✅ human_eco10.py HAS GENOME_START support!")
    else:
        print("❌ human_eco10.py does NOT have GENOME_LIMIT support")

print()
print("Checking a few more scripts...")

for script_name in ['human_eco17.py', 'human_fasta16.py', 'human_spiral8.py']:
    with open(script_name, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        has_support = 'GENOME_LIMIT' in content
        status = "✅" if has_support else "❌"
        print(f"{status} {script_name}: {'HAS' if has_support else 'NO'} GENOME_LIMIT support")
