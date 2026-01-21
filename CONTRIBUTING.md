# Contributing to scVAE-Annotator

Thank you for your interest in contributing to scVAE-Annotator! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Features

Feature suggestions are welcome! Please open an issue with:
- A clear description of the feature
- Use cases and benefits
- Any relevant examples or references

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format your code (`black src/ tests/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/or4k2l/scVAE-Annotator.git
cd scVAE-Annotator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install in development mode (optional):
```bash
pip install -e ".[dev]"
```

5. Test the pipeline:
```bash
python scvae_annotator.py
```

6. Run tests (if available):
```bash
pytest
```

## Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- We use [flake8](https://flake8.pycqa.org/) for linting
- Type hints are encouraged
- Docstrings should follow NumPy style

## Testing

- Write tests for all new features
- Maintain or improve code coverage
- Run tests before submitting PR: `pytest`
- Run with coverage: `pytest --cov=scvae_annotator`

## Documentation

- Update documentation for new features
- Include docstrings for all public functions/classes
- Add examples for complex features

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing!
