# Contributing to Neurolab

Thank you for your interest in contributing to **Neurolab** - the LIMINAL Heartbeat emotion recognition project! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this standard.

## How to Contribute

There are many ways to contribute to Neurolab:

- **Report bugs**: Found a bug? Open an issue with details
- **Suggest features**: Have an idea? Create a feature request
- **Improve documentation**: Help make our docs clearer
- **Write code**: Fix bugs or implement new features
- **Share research**: Contribute experiments or model improvements

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- Git

### Installation

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR-USERNAME/Neurolab.git
   cd Neurolab
   ```

3. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Install development dependencies**

   ```bash
   pip install pytest black flake8 mypy
   ```

6. **Create a branch for your work**

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Project Structure

```
Neurolab/
├── neurolab/               # Main package
│   ├── models/            # Model architectures
│   ├── data/              # Data loading utilities
│   ├── training/          # Training utilities
│   ├── utils/             # Helper functions
│   └── visualization/     # Visualization tools
├── configs/               # Configuration files
├── examples/              # Example scripts and notebooks
├── tests/                 # Unit tests
├── Osozn3.ipynb          # Original research notebook
├── demo.py               # Quick demo script
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
└── CONTRIBUTING.md       # This file
```

## Coding Standards

We follow PEP 8 style guidelines with some modifications:

### Code Formatting

- Use **Black** for automatic code formatting:
  ```bash
  black neurolab/
  ```

- Maximum line length: 100 characters

### Code Quality

- Use **flake8** for linting:
  ```bash
  flake8 neurolab/
  ```

- Use **mypy** for type checking:
  ```bash
  mypy neurolab/
  ```

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstrings:

  ```python
  def my_function(arg1: int, arg2: str) -> bool:
      """
      Brief description of function.

      Args:
          arg1 (int): Description of arg1
          arg2 (str): Description of arg2

      Returns:
          bool: Description of return value
      """
      pass
  ```

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

## Testing

We use **pytest** for testing.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=neurolab tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Example:

  ```python
  def test_model_forward_pass():
      model = TinyRecursiveModelTRMv6(dim=128)
      x = torch.randn(4, 128)
      y_init = torch.zeros(4, 128)

      y, confs, pad = model(x, y_init)

      assert y.shape == (4, 128)
      assert pad.shape == (4, 3)
      assert len(confs) == 5
  ```

## Submitting Changes

### Pull Request Process

1. **Ensure your code passes all checks**

   ```bash
   black neurolab/
   flake8 neurolab/
   pytest tests/
   ```

2. **Update documentation**
   - Update README.md if needed
   - Add docstrings to new code
   - Update CHANGELOG.md (if it exists)

3. **Commit your changes**

   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

   Use clear, descriptive commit messages:
   - `Add feature: emotion intensity prediction`
   - `Fix bug: incorrect PAD normalization`
   - `Improve docs: add training tutorial`
   - `Refactor: simplify attention mechanism`

4. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template with:
     - Description of changes
     - Related issue numbers
     - Testing performed
     - Screenshots (if applicable)

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Respond to review feedback promptly
- Ensure CI checks pass

## Reporting Issues

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, CUDA version)
- Error messages and stack traces
- Minimal code example

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (optional)
- Examples or mockups (if applicable)

## Model Contributions

If you're contributing a new model architecture:

1. Add the model to `neurolab/models/`
2. Follow the existing model structure
3. Include comprehensive docstrings
4. Add unit tests
5. Document in README.md
6. Provide example usage
7. Include training results/benchmarks

## Research Contributions

We welcome research contributions:

- New emotion models or theories
- Improved training techniques
- Novel evaluation metrics
- Dataset contributions
- Benchmark comparisons

Please document your methodology and results thoroughly.

## Questions?

- Open an issue for questions
- Check existing issues and PRs first
- Be respectful and patient

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Neurolab! 🧠❤️
