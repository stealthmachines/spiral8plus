# SARS-CoV-2 Genome Visualization Index

## Quick Start Guide

All visualization files are now ready to use with the SARS-CoV-2 Wuhan-Hu-1 genome!

### Requirements
```bash
pip install vispy pyqt6 numpy
```

### Run Any Visualization
```bash
python <filename>.py
```

---

## Visualization Categories

### 🌟 Recommended Highlights

#### Best Overall Visualizations
1. **`fasta17_auto.py`** - Enhanced multi-viral particle simulation with auto-detection
2. **`covid_spiral9.py`** - Infinite echoing φ-spiral with protein assemblies
3. **`covid8.py`** - Complete genome convergence to φ-core
4. **`covid9.py`** - Full virion (viral capsid) visualization

#### Best for Understanding Genome Structure
1. **`covid_spiral8.py`** - Closed geometries with inter-shape interactions
2. **`covid7.py`** - Double φ-spiral with echoes and protein links
3. **`covid_ecoli18.py`** - Advanced genome spiral visualization

---

## Complete File List (20 Files)

### FASTA Series (4 files)
Visual simulations with cellular/viral particle metaphors

| File | Description | Complexity |
|------|-------------|------------|
| `fasta17_auto.py` | Multi-viral particle simulation (auto-detect) | ⭐⭐⭐ |
| `fasta17_covid.py` | Multi-viral particle simulation (direct port) | ⭐⭐⭐ |
| `covid_fasta16.py` | Holographic multi-cell with division | ⭐⭐⭐ |
| `covid_fasta18.py` | Advanced FASTA visualization | ⭐⭐ |
| `covid_fasta19.py` | Enhanced FASTA rendering | ⭐⭐ |

### COVID Series (3 files)
Purpose-built COVID-19 visualizations

| File | Description | Complexity |
|------|-------------|------------|
| `covid7.py` | Double φ-spiral with echoes (continuous) | ⭐⭐ |
| `covid8.py` | Complete genome convergence (stops at end) | ⭐⭐ |
| `covid9.py` | Full virion with % completion tracker | ⭐⭐⭐ |

### Spiral Series (2 files)
Mathematical φ-spiral visualizations

| File | Description | Complexity |
|------|-------------|------------|
| `covid_spiral8.py` | φ-spiral with closed geometries | ⭐⭐ |
| `covid_spiral9.py` | Infinite echoing φ-spiral | ⭐⭐ |

### E.coli Converted Series (11 files)
Adapted from E. coli visualizations

| File | Original | Description |
|------|----------|-------------|
| `covid_ecoli10.py` | ecoli10.py | Genome visualization variant 10 |
| `covid_ecoli11.py` | ecoli11.py | Genome visualization variant 11 |
| `covid_ecoli12.py` | ecoli12.py | Genome visualization variant 12 |
| `covid_ecoli13.py` | ecoli13.py | Genome visualization variant 13 |
| `covid_ecoli14.py` | ecoli14.py | Genome visualization variant 14 |
| `covid_ecoli15.py` | ecoli15.py | Genome visualization variant 15 |
| `covid_ecoli18.py` | ecoli18.py | Advanced genome spiral |
| `covid_ecoli22.py` | ecoli22.py | Genome visualization variant 22 |
| `covid_ecoli23.py` | ecoli23.py | Genome visualization variant 23 |
| `covid_ecoli24.py` | ecoli24.py | Genome visualization variant 24 |
| `covid_ecoli25.py` | ecoli25.py | Genome visualization variant 25 |
| `covid_ecoli26.py` | ecoli26.py | Genome visualization variant 26 |
| `covid_ecoli27.py` | ecoli27.py | Genome visualization variant 27 |

---

## Feature Comparison

| Feature | fasta17_auto | covid_spiral8 | covid_spiral9 | covid8 | covid9 |
|---------|--------------|---------------|---------------|--------|--------|
| Auto-detect genome | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-particle | ✓ | ✗ | ✗ | ✗ | ✗ |
| Infinite loop | ✓ | ✓ | ✓ | ✗ | ✓ |
| Stops at end | ✗ | ✗ | ✗ | ✓ | ✗ |
| Progress % | ✓ | ✗ | ✓ | ✓ | ✓ |
| Protein echoes | ✓ | ✓ | ✓ | ✓ | ✓ |
| Inter-shape links | ✓ | ✓ | ✓ | ✓ | ✓ |
| Virion context | ✓ | ✗ | ✗ | ✗ | ✓ |

---

## Controls (All Visualizations)

- **Mouse drag:** Rotate camera view
- **Mouse wheel:** Zoom in/out
- **ESC:** Exit application

---

## Genome Information

**Current Data:**
- **Organism:** SARS-CoV-2 (COVID-19)
- **Isolate:** Wuhan-Hu-1 reference genome
- **Type:** Single-stranded RNA (+ssRNA)
- **Length:** 29,903 nucleotides
- **Assembly:** ASM985889v3
- **NCBI Reference:** NC_045512.2

**File Location:**
```
ncbi_dataset\data\GCF_009858895.2\GCF_009858895.2_ASM985889v3_genomic.fna
```

---

## Common Visualization Parameters

### Viral-Specific Adaptations
- **Core radius:** 12.0 (vs 15.0 for E. coli)
- **Strand separation:** 0.4 (vs 0.5)
- **Twist factor:** 3π (vs 2π)
- **Emergence speed:** Every 15-18 frames (vs 20)
- **Background:** Dark blue (#000011)

### Color Schemes
- **RNA strands:** Gold/yellow and cyan
- **Geometries:** Warm viral palette (red→orange→yellow→lime→cyan→blue→purple→magenta)
- **Protein links:** Golden/yellow (viral proteins)
- **Echoes:** Semi-transparent previous structures

---

## Performance Tips

### For Smooth Rendering:
1. Start with simpler visualizations (covid7.py, covid_spiral8.py)
2. Close other GPU-intensive applications
3. Reduce window size if needed
4. Lower MAX_POINTS in code if experiencing lag

### For Better Visuals:
1. Full-screen mode (F11 in some systems)
2. High-DPI displays show more detail
3. Darker room for better contrast
4. Let visualization run 2-3 minutes for full effect

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'vispy'"
```bash
pip install vispy pyqt6 numpy
```

### "FileNotFoundError: Could not find COVID-19 FASTA file"
- Verify genome file exists in `ncbi_dataset\data\`
- Check file paths in error message
- File should end with `.fna`

### Linter Warnings about "parent" parameter
- These are false positives
- VisPy does support the parent parameter
- Files will run correctly despite warnings

### Window appears but is blank
- Wait 2-3 seconds for initialization
- Check console for error messages
- Try a simpler visualization first

---

## Documentation Files

- **`COVID_BATCH_SUMMARY.md`** - Batch conversion summary
- **`COVID_VISUALIZATIONS.md`** - Detailed feature documentation
- **`README_FASTA17_COVID.md`** - FASTA17 variant documentation
- **`COVID_INDEX.md`** - This file

---

## Batch Processing Tools

### Conversion
```bash
python batch_convert_to_covid.py
```
Converts additional E. coli files to COVID versions

### Testing
```bash
python batch_test_covid.py
```
Validates all COVID visualization files

---

## Example Usage

### Quick Test
```bash
# Simplest visualization
python covid7.py
```

### Full Experience
```bash
# Most comprehensive visualization
python fasta17_auto.py
```

### Mathematical Beauty
```bash
# Infinite φ-spiral
python covid_spiral9.py
```

### Scientific Accuracy
```bash
# Complete genome convergence
python covid8.py
```

---

**Status:** ✓ All 20 files validated and ready
**Genome:** ✓ SARS-CoV-2 detected
**Dependencies:** Install with `pip install vispy pyqt6 numpy`
