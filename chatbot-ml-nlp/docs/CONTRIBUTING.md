# Contributing to ML/NLP Chatbot

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. Use the bug report template
3. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Relevant logs

### Suggesting Features

1. Check if the feature has been requested
2. Use the feature request template
3. Explain the use case and benefits
4. Provide implementation ideas if possible

### Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/chatbot-ml-nlp.git
   cd chatbot-ml-nlp
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set Up Development Environment**
   ```bash
   make install-dev
   pre-commit install
   ```

4. **Make Changes**
   - Write clean, documented code
   - Follow the style guide
   - Add tests for new features
   - Update documentation

5. **Run Tests**
   ```bash
   make test
   make lint
   ```

6. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `style:` formatting
   - `refactor:` code restructuring
   - `test:` adding tests
   - `chore:` maintenance

7. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub

## Development Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use meaningful variable names
- Add docstrings for functions and classes

### Testing

- Write unit tests for all new code
- Aim for >80% code coverage
- Test edge cases
- Use fixtures for common test data

### Documentation

- Update README if needed
- Add docstrings to functions
- Update API documentation
- Include examples for new features

## Project Structure

```
src/
├── nlp/              # NLP processing
├── classification/   # Intent classification
├── retrieval/        # Semantic search
└── ...
```

## Running Locally

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Start API server
make serve-api
```

## Need Help?

- Open an issue for questions
- Tag maintainers for urgent issues
- Check documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.