# Contributing to Human Genome Visualization Project

Thank you for your interest in contributing to the Human Genome Visualization Project! This document provides guidelines and information for contributors.

## 🚀 Ways to Contribute

### Code Contributions
- **Bug fixes**: Fix issues in existing visualization scripts
- **Performance improvements**: Optimize algorithms or add new acceleration methods
- **New visualizations**: Implement novel genome visualization approaches
- **Documentation**: Improve documentation, add examples, or fix errors

### Non-Code Contributions
- **Bug reports**: Report issues with clear reproduction steps
- **Feature requests**: Suggest new visualization techniques or improvements
- **Documentation**: Improve README, add tutorials, or create examples
- **Testing**: Test on different platforms or with different genome datasets

## 🛠️ Development Setup

### Prerequisites
- Python 3.7+
- Git
- Basic understanding of OpenGL/visualization concepts (helpful but not required)

### Setup Steps
1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/human-genome-visualization.git
   cd human-genome-visualization
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Download test data**:
   ```bash
   python genome_loader.py --ecoli  # Small dataset for testing
   ```
6. **Verify installation**:
   ```bash
   python genome_loader.py --verify
   ```

## 📝 Coding Standards

### Python Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use 4 spaces for indentation (no tabs)
- Line length: 88 characters maximum (Black formatter default)
- Use descriptive variable and function names

### Code Formatting
We use [Black](https://black.readthedocs.io/) for automatic code formatting:
```bash
pip install black
black .  # Format all Python files
```

### Type Hints
Use type hints for function parameters and return values:
```python
def calculate_phi_spiral(points: int, radius: float) -> np.ndarray:
    # Function implementation
    pass
```

### Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings:
```python
def genome_to_coordinates(sequence: str, scale: float = 1.0) -> np.ndarray:
    """Convert genome sequence to 3D coordinates using φ-spiral encoding.

    Args:
        sequence: DNA sequence string (ATCG bases)
        scale: Scaling factor for coordinate generation

    Returns:
        3D coordinates as numpy array of shape (n_bases, 3)

    Raises:
        ValueError: If sequence contains invalid bases
    """
```

## 🧪 Testing

### Running Tests
```bash
# Run basic verification
python genome_loader.py --verify

# Test individual scripts (with timeout to prevent hanging)
timeout 10 python human_eco1.py

# Run visualization tests
python -c "import vispy; import numpy as np; print('✓ Core imports work')"
```

### Test Data
- Use the E. coli genome for quick testing (`python genome_loader.py --ecoli`)
- For human genome testing, use environment variables to limit data:
  ```bash
  GENOME_LIMIT=10000 python your_script.py
  ```

## 📋 Pull Request Process

### Before Submitting
1. **Test your changes** thoroughly
2. **Update documentation** if needed
3. **Add examples** for new features
4. **Run code formatting**: `black .`
5. **Check linting**: `flake8 .` (if available)

### Pull Request Template
When creating a PR, please include:
- **Description**: What changes were made and why
- **Testing**: How the changes were tested
- **Screenshots**: For visualization changes (if applicable)
- **Performance**: Any performance implications
- **Breaking Changes**: If any existing functionality is affected

### Review Process
1. **Automated checks**: Code formatting and basic tests
2. **Peer review**: At least one maintainer review
3. **Testing**: Verification on multiple platforms if significant changes
4. **Merge**: Squash merge with descriptive commit message

## 🎨 Visualization Guidelines

### When Adding New Visualizations
1. **Follow naming conventions**: `human_[category][number].py`
2. **Include comprehensive comments** explaining the mathematical approach
3. **Support environment variables** for configuration
4. **Add to control panel** if appropriate
5. **Document in README** and script reference

### Mathematical Rigor
- **Cite sources** for mathematical formulas
- **Explain parameters** and their biological significance
- **Validate results** against known biological patterns
- **Document limitations** and assumptions

## 🐛 Reporting Bugs

### Bug Report Template
Please include:
- **Description**: Clear description of the issue
- **Reproduction steps**: Step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, hardware specs
- **Error messages**: Full traceback if applicable
- **Test data**: Which genome/dataset was used

### Debugging Tips
1. **Isolate the issue**: Test with minimal data (GENOME_LIMIT=1000)
2. **Check dependencies**: Run `python genome_loader.py --verify`
3. **GPU issues**: Try CPU-only mode if applicable
4. **Memory issues**: Monitor RAM usage with smaller datasets

## 📚 Documentation

### Updating Documentation
- **README.md**: Update for new features or installation changes
- **HUMAN_SCRIPTS_CONTROL_PANEL.md**: Add new scripts with descriptions
- **HUMAN_SCRIPTS_RATINGS.md**: Rate new scripts across all dimensions
- **Inline comments**: Keep code well-documented

### Documentation Standards
- Use Markdown for all documentation
- Include code examples where helpful
- Keep screenshots up to date
- Link to relevant scientific papers

## 🌍 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Maintain scientific integrity

### Communication
- Use GitHub Issues for bugs and feature requests
- Use GitHub Discussions for general questions
- Keep discussions technical and on-topic
- Be patient with responses

## 🎯 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Recognized for significant contributions
- Invited to join as maintainers for sustained contributions

## 📞 Getting Help

- **Documentation**: Check README.md and script references first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join related scientific communities

---

Thank you for contributing to advancing computational biology visualization! 🚀