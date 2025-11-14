# Analyze E. coli K-12 genome composition
seq = ''.join([line.strip() for line in open('ecoli_k12.fasta') if not line.startswith('>')]).upper()

print(f'Total length: {len(seq):,} bp')
print(f'\nNucleotide composition:')
print(f'A: {seq.count("A"):,} ({seq.count("A")/len(seq)*100:.1f}%)')
print(f'T: {seq.count("T"):,} ({seq.count("T")/len(seq)*100:.1f}%)')
print(f'G: {seq.count("G"):,} ({seq.count("G")/len(seq)*100:.1f}%)')
print(f'C: {seq.count("C"):,} ({seq.count("C")/len(seq)*100:.1f}%)')

print(f'\nFirst 100 bases: {seq[:100]}')
print(f'\nBase diversity in first 1000:')
print(f'  A={seq[:1000].count("A")} T={seq[:1000].count("T")} G={seq[:1000].count("G")} C={seq[:1000].count("C")}')

# Map to geometry dimensions (ecoli1.py mapping)
base_map = {'A': 5, 'T': 2, 'G': 4, 'C': 1}
geometry_names = {
    1: 'Point (red)',
    2: 'Line (green)',
    4: 'Tetrahedron (mediumpurple)',
    5: 'Pentachoron (blue)'
}

print(f'\n--- Geometry Mapping (ecoli1.py) ---')
for base in ['A', 'T', 'G', 'C']:
    dim = base_map[base]
    count = seq.count(base)
    print(f'{base} → Dim {dim}: {geometry_names[dim]:25} ({count:,} occurrences, {count/len(seq)*100:.1f}%)')

# Analyze sequential patterns
print(f'\n--- Sequential Pattern Analysis ---')
print(f'First 50 bases and their geometries:')
for i in range(50):
    base = seq[i]
    dim = base_map[base]
    print(f'{i:2d}: {base} → {geometry_names[dim]}')

# Calculate geometry transitions
print(f'\n--- Geometry Transition Statistics (first 10,000 bases) ---')
transitions = {}
for i in range(9999):
    b1, b2 = seq[i], seq[i+1]
    d1, d2 = base_map[b1], base_map[b2]
    key = f'{b1}({d1})→{b2}({d2})'
    transitions[key] = transitions.get(key, 0) + 1

for trans, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f'{trans:20} : {count:4d} times ({count/9999*100:.1f}%)')
