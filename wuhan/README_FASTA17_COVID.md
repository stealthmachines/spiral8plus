# FASTA17 COVID-19 Genome Visualizations

This folder contains new versions of fasta17.py that work with the SARS-CoV-2 Wuhan-Hu-1 genome data from the NCBI dataset.

## Files Created

### `fasta17_covid.py`
- Direct port of fasta17.py to use the COVID-19 genome
- Uses hardcoded path to the FASTA file in ncbi_dataset
- Optimized parameters for viral genome visualization
- Color-coded viral particles with unique identifiers

### `fasta17_auto.py` (Recommended)
- Automatically detects and loads the COVID-19 FASTA file
- Enhanced viral particle simulation
- Dynamic color generation for each viral particle
- Improved UI with progress tracking and statistics
- Adaptive visualization parameters for viral genomes

## Data Source
The FASTA files are located in:
```
ncbi_dataset\data\GCF_009858895.2\GCF_009858895.2_ASM985889v3_genomic.fna
ncbi_dataset\data\GCA_009858895.3\GCA_009858895.3_ASM985889v3_genomic.fna
```

## Genome Information
- **Organism**: Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2)
- **Isolate**: Wuhan-Hu-1 (reference genome)
- **Length**: ~29,903 nucleotides
- **Assembly**: ASM985889v3

## Requirements
```bash
pip install vispy pyqt6 numpy
```

## Usage
```bash
python fasta17_auto.py
```

## Visualization Features
- Real-time 3D genome spiral rendering
- Viral particle replication simulation
- Dynamic color coding for different particles
- Protein/organelle clustering visualization
- Interactive camera controls (mouse rotation, zoom)
- Progress tracking and statistics

## Key Differences from Original fasta17.py
1. **Shorter genome**: COVID-19 (~30K bases) vs E. coli (~4.6M bases)
2. **Viral characteristics**: Faster replication, different spiral parameters
3. **Enhanced visualization**: Better color coding, UI improvements
4. **Auto-detection**: Automatically finds the correct FASTA file
5. **Clustering behavior**: Viral particles exhibit clustering dynamics

## Controls
- **Mouse**: Rotate view
- **Mouse wheel**: Zoom in/out
- **ESC**: Exit application