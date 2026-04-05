# Contributing to Multimodal RAG System

Thank you for your interest in contributing to the Multimodal RAG System for Sustainability Documents! We welcome contributions from the community.

## How to Contribute

1. **Fork the repository** and create your feature branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Ensure tests pass** and code lints cleanly
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description of your changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/Bootcamp-Assignment-Multimodal-RAG-2024TM05010.git
cd Bootcamp-Assignment-Multimodal-RAG-2024TM05010

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest test_system.py -v

# Start development server
uvicorn app.main:app --reload
```

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write comprehensive docstrings
- Keep functions focused and modular
- Add unit tests for all new functionality

## Areas for Contribution

- **PDF Processing**: Improve multimodal document parsing
- **VLM Integration**: Enhance image analysis capabilities
- **RAG Pipeline**: Add new query types and improve retrieval
- **API Development**: Extend REST endpoints and add new features
- **Testing**: Increase test coverage and add integration tests
- **Documentation**: Improve README, API docs, and code comments

## Reporting Issues

When reporting bugs or requesting features, please:

- Use a clear and descriptive title
- Provide detailed steps to reproduce the issue
- Include relevant code snippets or error messages
- Specify your environment (OS, Python version, etc.)

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.