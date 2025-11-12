# COVID-19 Genome Visualization Files

This directory contains SARS-CoV-2 Wuhan-Hu-1 genome visualization scripts adapted from the original E. coli versions.

## Files Overview

### Original → COVID Adaptations

| Original File | COVID Version | Description |
|--------------|---------------|-------------|
| `ecoli7.py` | `covid7.py` | Double φ-spiral with rungs, echoes, and inter-links (continuous loop) |
| `ecoli8.py` | `covid8.py` | Complete genome convergence to φ-core (stops at end) |
| `ecoli9.py` | `covid9.py` | Full virion visualization with % completion tracker |
| `fasta17.py` | `fasta17_covid.py` | Multi-cell viral particle simulation |
| `fasta17.py` | `fasta17_auto.py` | Enhanced auto-detecting viral simulation |

## Key Adaptations for SARS-CoV-2

### 1. Genome Size
- **E. coli K-12**: ~4,641,652 bp (DNA)
- **SARS-CoV-2**: ~29,903 nt (RNA)
- Visualization parameters optimized for shorter viral genome

### 2. RNA vs DNA
- Base mapping includes 'U' (uracil) for RNA viruses
- T treated as U in coronavirus files
- Single-stranded RNA visualization in `covid9.py`

### 3. Viral Structure Parameters
```python
# Viral-specific settings:
core_radius = 12.0       # Smaller (vs 15.0 for E. coli)
strand_sep = 0.4         # Tighter (vs 0.5)
twist_factor = 3*np.pi   # More twist (vs 2*np.pi)
virion_length = 15.0     # Compact viral capsid
virion_radius = 4.0      # Small viral particle
```

### 4. Visual Enhancements
- **Background**: Dark blue (`#000011`) for better contrast
- **Strand colors**: Gold/yellow and cyan for RNA strands
- **Enhanced protein links**: Golden color for viral proteins
- **Progress tracking**: Real-time nucleotide count and percentage

### 5. Animation Speeds
- Faster emergence: every 15 frames (vs 20)
- More visible echoes: alpha 0.3 (vs 0.25)
- Adjusted camera movements for viral scale

## Usage

All files auto-detect the SARS-CoV-2 FASTA file from:
```
ncbi_dataset\data\GCF_009858895.2\GCF_009858895.2_ASM985889v3_genomic.fna
ncbi_dataset\data\GCA_009858895.3\GCA_009858895.3_ASM985889v3_genomic.fna
```

### Requirements
```bash
pip install vispy pyqt6 numpy
```

### Run
```bash
python covid7.py    # Continuous loop with echoes
python covid8.py    # Convergence to completion
python covid9.py    # Full virion with % tracker
```

## Features Comparison

### covid7.py
✅ Continuous genome loop
✅ Inter-rung protein links
✅ Echo visualization
✅ Real-time progress %

### covid8.py
✅ Runs until complete genome convergence
✅ Stops at nucleotide 29,903
✅ Final φ-core convergence
✅ Completion indicator

### covid9.py
✅ Full virion (viral capsid) context
✅ Single-stranded RNA visualization
✅ Dynamic progress color coding
✅ Viral protein-RNA interactions
✅ Compact viral particle structure

## Scientific Accuracy

These visualizations use:
- **Golden ratio (φ)**: Mathematical elegance in genomic spiral
- **Base-specific geometry**: Each nucleotide mapped to unique 3D shape
- **φ-spiral encoding**: 137.507° golden angle for optimal packing
- **Viral compaction**: Parameters reflecting actual viral genome density

## Controls

All visualizations support:
- **Mouse drag**: Rotate view
- **Mouse wheel**: Zoom in/out
- **ESC**: Exit application

## Data Source

NCBI Reference Sequence: NC_045512.2
- Organism: Severe acute respiratory syndrome coronavirus 2
- Isolate: Wuhan-Hu-1
- Assembly: ASM985889v3
- Complete genome: 29,903 nucleotides
