# Human Genome Visualization Control Panel

**Interactive launcher for all 80 human genome visualization scripts**

## Quick Start

```powershell
python human_genome_control_panel.py
```

## What It Does

The control panel provides an interactive menu to:

1. **Select Number of Nucleotides** - Choose how much genome data to load
2. **Select Chromosome/Region** - Pick which part of the genome to visualize
3. **Select Visualization Script** - Choose from 80 different visualization styles
4. **Launch with One Click** - Automatically sets environment variables and runs the script

## Features

### 📊 Nucleotide Selection
- **Quick Preview** (10,000) - Instant loading
- **Standard View** (100,000) - Fast, recommended
- **Detailed View** (500,000) - Medium speed
- **Extended View** (1,000,000) - Slower
- **Chromosome View** (10,000,000) - Very slow
- **Full Genome** (3.1 billion) - Not recommended
- **Custom Amount** - Enter any number

### 🧬 Chromosome Selection
- Chromosome 1 (largest, 249M bases)
- Chromosome 2 (242M bases)
- Chromosome X (154M bases)
- Chromosome Y (57M bases)
- Mitochondrial DNA (16,569 bases)
- Start from beginning (default)
- Custom chromosome ID
- Custom start position

### 🎨 Script Categories

**80 Total Scripts:**
- **50 eco-based** visualizations (human_eco*.py)
- **28 Fasta-based** visualizations (human_fasta*.py)
- **2 Spiral** visualizations (human_spiral*.py)

## Usage Examples

### Example 1: Quick Preview
```
Step 1: Select "Quick Preview (10,000 nucleotides)"
Step 2: Select "Start from beginning"
Step 3: Select any script (e.g., human_eco17.py)
Launch: yes
```

### Example 2: Chromosome-Specific
```
Step 1: Select "Standard View (100,000 nucleotides)"
Step 2: Select "Chromosome X"
Step 3: Select visualization script
Launch: yes
```

### Example 3: Custom Region
```
Step 1: Select "Custom amount" → Enter: 50000
Step 2: Select "Custom start position" → Enter: 1000000
Step 3: Select script
Launch: yes
```

## Environment Variables Set

The control panel automatically sets these environment variables when launching:

```powershell
GENOME_LIMIT=100000              # Number of nucleotides
GENOME_CHROMOSOME=NC_000001.11   # Optional: specific chromosome
GENOME_START=0                   # Optional: start position
```

## Manual Launch (Without Control Panel)

If you prefer to launch scripts directly:

```powershell
# Standard view from beginning
$env:GENOME_LIMIT="100000"; python human_eco17.py

# Specific chromosome
$env:GENOME_LIMIT="100000"; $env:GENOME_CHROMOSOME="NC_000001.11"; python human_fasta16.py

# Custom start position
$env:GENOME_LIMIT="50000"; $env:GENOME_START="1000000"; python human_spiral8.py
```

## Script Compatibility

### Scripts WITH Full Control Panel Support
These scripts support all features (GENOME_LIMIT, GENOME_CHROMOSOME, GENOME_START):
- human_eco17.py through human_eco48.py
- human_spiral8.py, human_spiral9.py

### Scripts WITH Partial Support
These scripts support GENOME_LIMIT only:
- Most human_fasta*.py files
- Some human_eco*.py files

### Scripts WITHOUT Environment Variable Support
These scripts load the entire genome (slower):
- Legacy scripts without GENOME_LIMIT

**Note:** Control panel works with all scripts, but some may not respect all settings.

## Requirements

- Python 3.7+
- Human genome FASTA file installed at:
  - `ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna`
  - OR `ncbi_dataset\data\GCF_000001405.40\*.fna`

## Tips

### For Best Performance
1. Start with "Quick Preview" (10,000 nucleotides)
2. Use "Standard View" for most visualizations
3. Avoid "Full Genome" unless you have 16GB+ RAM

### For Specific Research
1. Select the chromosome you're interested in
2. Use custom start position to focus on a gene region
3. Load just enough nucleotides to cover your area of interest

### For Exploration
1. Try different visualization scripts with the same data
2. Compare eco vs fasta vs spiral visualizations
3. Use "list" command to see all available scripts

## Troubleshooting

### "No scripts found"
- Make sure you're in the advanced-spiral-8 directory
- Check that human_*.py files exist

### "FASTA file not found"
- Download human genome data first
- Check paths in script match your installation

### Script exits immediately
- Some scripts require VisPy: `pip install vispy pyqt6 numpy`
- Check that genome data is accessible

### Out of memory
- Reduce nucleotide count
- Use Quick or Standard view instead of Full Genome

## Advanced Features

### Browse by Category
At the script selection prompt, type:
- `eco` - Show all 50 eco visualizations
- `fasta` - Show all 28 fasta visualizations
- `list` - Show all 80 scripts

### Run Multiple Visualizations
After a visualization completes, you'll be asked:
```
Run another visualization? (yes/no):
```

Type `yes` to launch another script with different settings.

## Examples by Use Case

### Genomic Research
```
Purpose: Study first 1 million bases of Chromosome 1
Settings:
  - Nucleotides: 1,000,000
  - Chromosome: NC_000001.11 (Chromosome 1)
  - Script: human_eco17.py (has GENOME_LIMIT support)
```

### Quick Demo
```
Purpose: Fast visualization demo
Settings:
  - Nucleotides: 10,000
  - Region: Start from beginning
  - Script: human_spiral8.py (3D spiral visualization)
```

### Gene-Specific Visualization
```
Purpose: Visualize region around specific gene
Settings:
  - Nucleotides: 50,000
  - Start position: [gene start position]
  - Script: Any with position support
```

## Control Panel Features

### Smart Discovery
- Automatically finds all human_*.py scripts
- Categorizes by type (eco/fasta/spiral)
- Shows file count for each category

### User-Friendly Interface
- Clear step-by-step prompts
- Input validation
- Helpful error messages
- Configuration summary before launch

### Flexible Selection
- Browse by category
- Show full list
- Quick selection by number
- Back navigation

## Exit Options

- Type `no` at any confirmation to cancel
- Press `Ctrl+C` during visualization to stop
- Type `no` when asked to run another visualization

---

**Created:** November 11, 2025
**Total Scripts:** 80 human genome visualizations
**Author:** Control Panel System
**Status:** Production Ready ✅
