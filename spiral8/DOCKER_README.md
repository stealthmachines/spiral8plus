# Grand Unified Theory - Docker Testing Environment
# =================================================

## Quick Start

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 8+ GB RAM
- 10+ GB disk space

### Build and Test (Windows)
```powershell
.\docker_build.ps1
```

### Build and Test (Linux/Mac)
```bash
chmod +x docker_build.sh
./docker_build.sh
```

## Manual Docker Commands

### Build Image
```bash
docker build -t gut-testing:latest .
```

### Run Main Validation
```bash
docker run --rm gut-testing python grand_unified_theory.py
```

### Run Interactive Demo
```bash
docker run --rm -v ${PWD}/plots:/gut/plots gut-testing python gut_demo.py
```

### Run C Precision Engine
```bash
docker run --rm gut-testing gut_engine validate-all
```

### Download LIGO Data
```bash
docker run --rm -it -v ${PWD}/ligo_data:/gut/ligo_data gut-testing python download_data.py
```

### Run Data Analysis (after downloading LIGO data)
```bash
docker run --rm -v ${PWD}/output:/gut/output -v ${PWD}/ligo_data:/gut/ligo_data gut-testing python gut_data_analysis.py
```

### Interactive Shell
```bash
docker run --rm -it gut-testing /bin/bash
```

## Docker Compose (Multi-Service)

### Run All Services
```bash
docker-compose up
```

### Run Specific Service
```bash
docker-compose run gut-test
docker-compose run gut-demo
docker-compose run gut-analysis
docker-compose run gut-c-engine
```

### Run with Jupyter Notebook
```bash
docker-compose --profile interactive up gut-notebook
```
Then open http://localhost:8888 in your browser.

### Stop All Services
```bash
docker-compose down
```

## Data Volumes

The Docker setup mounts these directories:
- `./bigG` → Pan-STARRS supernova data (read-only)
- `./micro-bot-digest` → Micro-scale symbolic fits (read-only)
- `./hdgl_harmonics_spiral10000_analog_v30` → HDGL data (read-only)
- `./output` → Generated reports (read-write)
- `./plots` → Generated plots (read-write)
- `./logs` → Log files (read-write)
- `./ligo_data` → Downloaded LIGO data (read-write)

## Environment Variables

Set these in `docker-compose.yml` or with `-e` flag:

```bash
GUT_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
GUT_PRECISION=high          # standard, high, extreme
GUT_ANALYSIS_MODE=full      # quick, standard, full
GUT_PLOT_FORMAT=png         # png, pdf, svg
OPENBLAS_NUM_THREADS=4      # Number of threads for linear algebra
OMP_NUM_THREADS=4           # Number of OpenMP threads
```

Example:
```bash
docker run --rm -e GUT_LOG_LEVEL=DEBUG -e GUT_PRECISION=extreme gut-testing
```

## Troubleshooting

### Out of Memory
Increase Docker Desktop memory limit:
- Windows/Mac: Docker Desktop → Settings → Resources → Memory → 8+ GB

### Permission Errors (Linux)
Run with user permissions:
```bash
docker run --rm --user $(id -u):$(id -g) -v ${PWD}/output:/gut/output gut-testing
```

### Slow Build
Use BuildKit for faster builds:
```bash
DOCKER_BUILDKIT=1 docker build -t gut-testing .
```

### Network Issues During Build
Use a mirror or proxy:
```bash
docker build --network=host -t gut-testing .
```

## Performance Optimization

### Multi-Core Testing
```bash
docker run --rm --cpus=4 gut-testing python grand_unified_theory.py
```

### GPU Support (NVIDIA)
Install nvidia-docker2, then:
```bash
docker run --rm --gpus all gut-testing python animate_full_waterfall_gpu12.py
```

### Reduce Image Size
```bash
docker image prune -a  # Remove unused images
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: GUT Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t gut-testing .
      - name: Run tests
        run: docker run gut-testing python grand_unified_theory.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: gut-report
          path: output/gut_report.json
```

## Data Flow

```
Input Data (Local)
  ├─ bigG/                     → Supernova analysis
  ├─ micro-bot-digest/         → Micro-scale validation
  └─ hdgl_harmonics.../        → High-precision tests
       ↓
Docker Container
  ├─ grand_unified_theory.py   → Main framework
  ├─ gut_data_analysis.py      → Data validation
  ├─ gut_precision_engine (C)  → High-precision compute
  └─ gut_demo.py               → Interactive demo
       ↓
Output Data (Local)
  ├─ output/gut_report.json    → Validation results
  ├─ plots/*.png                → Visualizations
  └─ logs/*.log                 → Execution logs
```

## Testing Strategy

### 1. Quick Smoke Test (30 seconds)
```bash
docker run --rm gut-testing python gut_demo.py
```

### 2. Full Validation (5 minutes)
```bash
docker run --rm -v ${PWD}/output:/gut/output gut-testing python grand_unified_theory.py
```

### 3. Real Data Analysis (30 minutes)
```bash
# Download LIGO data first
docker run --rm -it -v ${PWD}/ligo_data:/gut/ligo_data gut-testing python download_data.py

# Run full analysis
docker run --rm -v ${PWD}/output:/gut/output -v ${PWD}/ligo_data:/gut/ligo_data gut-testing python gut_data_analysis.py
```

### 4. C Precision Engine (10 minutes)
```bash
docker run --rm gut-testing gut_engine validate-all
```

## Expected Results

### Successful Output
```
✓ Planck units: 0.00% error
✓ Dark energy: 0.13% error (5.952e-10 vs 5.960e-10 J/m³)
✓ QNM spectrum: φ-harmonic series confirmed
✓ Cross-scale consistency: PASS
```

### Key Predictions to Test
1. **Black Hole φ-Echoes**: 3.44% amplitude at ~44 μs delay
2. **QNM Frequency Ratios**: f_n = f₀ · φⁿ (not f_n = f₀ · 1.5ⁿ as in GR)
3. **Dark Energy Density**: ρ_Λ = 5.952×10⁻¹⁰ J/m³

## Cleanup

### Remove All Containers
```bash
docker-compose down -v
```

### Remove Images
```bash
docker rmi gut-testing:latest
```

### Remove All GUT Data
```bash
docker system prune -a
```

## Next Steps

1. **Download Real Data**: `python download_data.py`
2. **Analyze LIGO Events**: Test φ-echo predictions
3. **Refine Parameters**: Optimize micro-scale accuracy
4. **Publish Results**: Generate publication-ready figures

## Support

For issues or questions:
- Check `logs/*.log` for error messages
- Verify data inventory: `python data_inventory.py`
- Run diagnostics: `docker run --rm gut-testing gut_engine diagnose`

## License

Research use only. See project README for details.
