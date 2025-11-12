# Script Compatibility Guide

## Environment Variable Support

### ✅ Full Support (65+ scripts)
These scripts support ALL environment variables: GENOME_LIMIT, GENOME_CHROMOSOME, GENOME_START

**Spiral Scripts (2 scripts):**
- `human_spiral8.py` ✅
- `human_spiral9.py` ✅

**Eco Scripts (43 scripts with load_genome):**
- `human_eco1.py`, `human_eco10.py` through `human_eco48.py` (excluding eco2, eco4) ✅
- `human_eco_unified_phi_synthesis.py` ✅

**Fasta Scripts (23 scripts with load_genome):**
- `human_fasta1.py` through `human_fasta9.py` (excluding fasta10-13) ✅
- `human_fasta14.py` through `human_fasta19.py` ✅
- `human_fasta17_auto.py`, `human_fasta17_covid.py` ✅
- `human_fasta4b.py` through `human_fasta4g.py` ✅

**Features:**
- Respects GENOME_LIMIT (limits nucleotides loaded)
- Respects GENOME_CHROMOSOME (loads specific chromosome)
- Respects GENOME_START (starts from custom position)
- Fast loading with limits
- **NEW: Now works great with control panel!**

### ⚠️ Limited/No Support (15 scripts)
These scripts use different loading mechanisms or don't have load_genome function

**Eco Scripts without load_genome (9 scripts):**
- `human_eco.py`, `human_eco2.py`, `human_eco4.py`
- `human_eco46_c_engine.py`
- `human_eco46_v2_100percent_fasta.py`
- `human_eco46_v3_ai_interpreter.py`
- `human_eco46_v3_gpu_full.py`
- `human_eco46_v3_pure_fasta.py`
- `human_eco46_v3_terminal.py`

**Fasta Scripts without load_genome (6 scripts):**
- `human_fasta2GPU.py`
- `human_fasta10.py` through `human_fasta13.py`

**Behavior:**
- May use different loading mechanisms (C engine, GPU, etc.)
- Environment variables may not be supported
- Use with caution

## Recommendations

### For Control Panel Use
**Best Choice:** Most scripts now work great! 65+ scripts support all environment variables.
- ✅ All spiral scripts (2)
- ✅ Most eco scripts (43 out of 50)
- ✅ Most fasta scripts (23 out of 28)
- Fast loading with GENOME_LIMIT
- Works with chromosome selection
- Works with custom start positions

### Recommended Scripts for Quick Testing
- `human_spiral8.py` or `human_spiral9.py` - Always work, 3D spiral visualization
- `human_eco17.py` - Popular eco visualization
- `human_fasta16.py` - FASTA-based visualization

### For Direct Command Line Use
Now works great with most scripts!

```powershell
# Quick preview with eco script (NOW WORKS!)
$env:GENOME_LIMIT="10000"
python human_eco17.py  # Loads only 10K nucleotides ✅

# Chromosome-specific loading (NOW WORKS!)
$env:GENOME_LIMIT="100000"; $env:GENOME_CHROMOSOME="NC_000001.11"
python human_fasta16.py  # Loads 100K from Chromosome 1 ✅

# Custom start position (NOW WORKS!)
$env:GENOME_LIMIT="50000"; $env:GENOME_START="1000000"
python human_eco20.py  # Loads 50K starting at position 1M ✅
```

### Scripts Without Support (15 total)
These scripts use special loading mechanisms and may not respect environment variables:
- C engine variants
- GPU-accelerated variants
- Some early versions without load_genome function

### Adding Support to Eco/Fasta Scripts

To add GENOME_LIMIT support to an eco/fasta script, add this code before loading:

```python
import os

# Add at the top with imports
def load_genome(fasta_file, max_nucleotides=None, chromosome=None):
    """
    Load genome sequence from FASTA file

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var)
        chromosome: Specific chromosome to load (None = use GENOME_CHROMOSOME env var)
    """
    # Get from environment if not specified
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        try:
            max_nucleotides = int(env_limit)
        except ValueError:
            max_nucleotides = 100000  # Default

    if chromosome is None:
        chromosome = os.environ.get('GENOME_CHROMOSOME', None)

    sequence = ""
    current_chromosome = None
    nucleotide_count = 0

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                # New chromosome header
                current_chromosome = line.strip()[1:].split()[0]

                # If we're filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                print(f"Loading from {current_chromosome}...")
            else:
                # If filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                # Add nucleotides up to limit
                bases = line.strip()
                remaining = max_nucleotides - nucleotide_count

                if remaining <= 0:
                    break

                sequence += bases[:remaining]
                nucleotide_count += len(bases[:remaining])

                if nucleotide_count >= max_nucleotides:
                    print(f"Reached limit of {max_nucleotides:,} nucleotides")
                    break

    return sequence
```

## Control Panel Workflow

When you use the control panel:

1. **Control Panel Sets Environment Variables:**
   ```python
   env['GENOME_LIMIT'] = '10000'
   env['GENOME_CHROMOSOME'] = 'NC_000001.11'
   ```

2. **Script Behavior:**
   - **Spiral scripts:** ✅ Read environment variables, load only requested data
   - **Eco/Fasta scripts:** ⚠️ Ignore environment variables, load entire genome

3. **Result:**
   - Spiral scripts: Fast (seconds)
   - Eco/Fasta scripts: Very slow (minutes) or out of memory

## Quick Test

Test which scripts support environment variables:

```powershell
# Test spiral script (SHOULD work fast)
$env:GENOME_LIMIT="10000"
python human_spiral8.py
# Result: Loads only 10,000 nucleotides ✅

# Test eco script (will be SLOW)
$env:GENOME_LIMIT="10000"
python human_eco10.py
# Result: Tries to load entire genome ⚠️
```

## Summary

| Script Type | Count | GENOME_LIMIT | GENOME_CHROMOSOME | GENOME_START | Speed |
|------------|-------|--------------|-------------------|--------------|-------|
| Spiral | 2 | ✅ | ✅ | ✅ | Fast |
| Eco (with load_genome) | 43 | ✅ | ✅ | ✅ | Fast |
| Fasta (with load_genome) | 23 | ✅ | ✅ | ✅ | Fast |
| Special variants | 15 | ⚠️ | ⚠️ | ⚠️ | Varies |

**Total with Full Support: 68 out of 80 scripts (85%)**

**Recommendation:** 🎉 Most scripts now work perfectly with the control panel!
