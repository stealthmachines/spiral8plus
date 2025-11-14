# Readme for ‘Human Genome Only’ (a similar process for other builds)

# Human Genome Visualization Project
[![GitHub stars](https://img.shields.io/github/stars/stealthmachines/spiral8plus.svg)](https://github.com/stealthmachines/spiral8plus/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/stealthmachines/spiral8plus.svg)](https://github.com/stealthmachines/spiral8plus/network)
[![License: Custom](https://img.shields.io/badge/License-Custom-lightgrey.svg)](LICENSE.txt)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

> **Revolutionary genome visualization through φ-spiral encodings and pure genome-driven mathematics**

This project implements groundbreaking approaches to genome visualization, featuring:

* **Pure Genome Mathematics**: Zero arbitrary constants in advanced implementations
* **φ-Framework Integration**: Complete physical constants integration (CODATA)
* **Real-time GPU Acceleration**: Interactive visualization of million-point datasets
* **Emergent Behavior**: Self-organizing systems from genome data alone

## 🌟 Key Features

* **70+ Visualization Scripts**: From basic φ-spirals to universe-scale simulations
* **Multiple Performance Tiers**: CPU, GPU, and C-engine acceleration (100x+ speedup)
* **Comprehensive Documentation**: Comprehensive technical references
* **Cross-Platform**: Windows, Linux, and macOS support
* **Research-Grade**: Based on phyllotaxis mathematics and emergent physical constants as developed by [zchg.org](https://zchg.org)

## 🚀 Quick Start

### 1. Clone or Download the Repository

```bash
git clone https://github.com/stealthmachines/spiral8plus.git
cd spiral8plus
```

### 2. Install Dependencies

```bash
# Install Python requirements
pip install numpy vispy pyqt6

# Optional but perhaps recommended
pip install biopython

# For C-engine acceleration (Windows)
# Run build_engine.bat or build_engine_v2.bat
```

### 3. Download Genome Data

**⚠️ Important**: Genome data files are large and cannot be included in the repository.  You will need to download the FASTA or FNA files separately and place them in your data folder (see the file structure below, for example):

E. Coli - [https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3?report=fasta](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3?report=fasta)

Human - [https://www.ncbi.nlm.nih.gov/nuccore/HQ287898.1?report=fasta](https://www.ncbi.nlm.nih.gov/nuccore/HQ287898.1?report=fasta)

### 4. Run Visualizations

The Control Panel is very helpful, but only works for human genome at this time - /advanced-spiral-8- Human Genomes/human_genome_control_panel.py

```bash
# Interactive control panel
python human_genome_control_panel.py

# Individual scripts
python human_eco46_v2_100percent_fasta.py

# With environment variables
GENOME_LIMIT=50000 GENOME_CHROMOSOME=NC_000001.11 python human_fasta19.py
```

## 📁 Project Structure (‘Human Genomes’ folder, in this example)

```
/advanced-spiral-8- Human Genomes/
├── genome_loader.py              # Automated genome data downloader
├── human_genome_control_panel.py # Interactive script launcher
├── requirements.txt              # Python dependencies
├── \ncbi_dataset\ncbi_dataset\data  # Genome data (downloaded, not committed)
│   ├── "human.fasta or FNA Files"              # Human genome reference (default)
│   ├── "ecoli.fasta or FNA Files"             # E. coli genome
│   └── "yeast.fasta or FNA Files"              # Yeast genome
├── HUMAN_SCRIPTS_CONTROL_PANEL.md # Complete script reference
├
├
└── human_*.py                    # 70+ visualization scripts
    ├── human_eco*.py            # φ-spiral encodings (50+ scripts)
    ├── human_fasta*.py          # Genome-driven visualizations (19+ scripts)
    ├── human_spiral*.py         # Closed geometry spirals (2 scripts)
    └── human_*.py               # Advanced analysis frameworks
```

Please note: at this time human_genome_control_panel.py is only for human genomes.  For all other cases, you will need to run the python files individually.

## 🎨 Visualization Categories

### Eco Series (φ-Spiral Encodings)

GPU-accelerated φ-spiral encodings with double strands, rungs, echoes, and convergence metrics.

**Highlights:**

* `human_eco46_v2_100percent_fasta.py`: Pure genome-driven, zero arbitrary constants
* `human_eco_unified_phi_synthesis.py`: Complete φ-framework with CODATA constants
* `human_eco46_c_engine.py`: 100x+ performance with C backend

### FASTA Series (Genome-Driven)

Holographic visualizations where all parameters emerge from FASTA sequences.

**Highlights:**

* `human_fasta19.py`: Ultra performance (1M+ points, 200 cells)
* `human_fasta14.py`: GPU-accelerated batch updates
* `human_fasta4c.py`: Food system with mutation simulation

### Advanced Frameworks

Specialized mathematical analysis using phi-frameworks.

**Highlights:**

* `human_eight_geometries_phi.py`: 8-dimensional geometric analysis
* `human_cross_cavity_tuning.py`: Resonance pattern analysis
* `human_cubic_scaling_analysis.py`: Scaling law applications

## 🔧 Environment Variables

Configure visualization behavior:

```bash
# Limit nucleotides loaded (performance control)
export GENOME_LIMIT=100000

# Specific chromosome (human genome)
export GENOME_CHROMOSOME=NC_000001.11

# Starting position in sequence
export GENOME_START=0
```

## 📚 Documentation

### Complete References

* **[Script Reference](HUMAN_SCRIPTS_CONTROL_PANEL.md)**: Detailed descriptions of all 70+ scripts
* **[Script Ratings](HUMAN_SCRIPTS_RATINGS.md)**: Quantitative evaluation across 9 dimensions
* **[Repository Rating](REPOSITORY_RATING.md)**: Comprehensive project assessment

### Key Documentation Files

* `HUMAN_SCRIPTS_CONTROL_PANEL.md`: Complete technical reference
* `HUMAN_SCRIPTS_RATINGS.md`: Performance and quality ratings
* `REPOSITORY_RATING.md`: Overall project assessment

## 🏗️ Building C Engines (Performance)

For maximum performance, build the C acceleration engines:

### Windows

```batch
build_engine.bat
build_engine_v2.bat
build_engine_v3.bat
```

### Linux/macOS

```bash
chmod +x build.sh
./build.sh
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
python genome_loader.py --verify
python -c "import vispy, numpy; print('Core dependencies OK')"
```

Test individual scripts:

```bash
timeout 10 python human_eco1.py  # Should show visualization briefly
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Coding Standards

* Follow PEP 8 style guidelines
* Add docstrings to new functions
* Update documentation for significant changes
* Test on multiple platforms when possible

## 📄 License

This project is licensed under a custom license - see the [LICENSE.txt](LICENSE.txt) file for details or visit [https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440](https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440)

**Important Note**: This software is provided for research and educational purposes. Genome data downloaded using the genome loader comes from public NCBI databases and is subject to NCBI’s terms of use.

## 🙏 Acknowledgments

* **NCBI**: For providing comprehensive genome databases
* **VisPy**: For the excellent OpenGL visualization framework
* **NumPy**: For high-performance numerical computing
* **Biopython**: For FASTA file parsing capabilities

## 📞 Support

* **Issues**: [GitHub Issues](https://github.com/stealthmachines/spiral8plus/issues)
* **Discussions**: https://zchg.org/t/human-genome-grch38-p14-visualization-using-zchg-orgs-tech/877/3/
* **Documentation**: See [HUMAN_SCRIPTS_CONTROL_PANEL.md](HUMAN_SCRIPTS_CONTROL_PANEL.md)

## 🔬 Research & Citation

This work represents cutting-edge research in computational biology visualization. If you use this software in your research, please cite:

```bibtex
@software{human_genome_visualization,
  title = {zCHG.org Spiral8 Plus Human Genome Visualization Project},
  author = {Josef Kulovany},
  url = {https://github.com/stealthmachines/spiral8plus},
  year = {2025},
  note = {Revolutionary genome visualization through φ-spiral encodings}
}
```

### Key Innovations

* **Pure Genome Mathematics**: Zero arbitrary constants in visualization algorithms
* **φ-Framework Integration**: Complete physical constants (CODATA) in biological context
* **Emergent Behavior**: Self-organizing systems from genome data alone
* **Performance Breakthroughs**: 100x+ speedup with C engine integration

## 🚀 Future Roadmap

* **Web Interface**: Browser-based genome exploration
* **VR/AR Support**: Immersive genome visualization
* **Multi-Omics Integration**: Combined genomics/proteomics/transcriptomics
* **AI/ML Enhancement**: Machine learning for pattern discovery
* **Quantum Computing**: Next-generation computational approaches

---

**⭐ SHARE this repository if you find it useful!  It could save lives…**

*Built for advancing computational biology and mathematical visualization under hybrid-distributed stewardship*

Special Thanks: https://www.ncbi.nlm.nih.gov/nuccore/
