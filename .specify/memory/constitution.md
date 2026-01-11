# NurseSim-RL Project Constitution

*Following spec-kit principles for intent-driven development*

## Code Quality Principles

### Type Safety & Documentation
- **Type Hints**: All function signatures must include type hints for parameters and return values
- **Docstrings**: Every public function, class, and module must have comprehensive docstrings following Google style
- **Error Handling**: Explicit error handling with informative messages; never silent failures

### Testing Standards
- **Unit Tests**: All Gymnasium environment methods must have unit tests with >80% coverage
- **Integration Tests**: End-to-end tests for agent task processing (Gradio and A2A modes)
- **Regression Tests**: Existing tests must pass before any commit to main branch
- **Test Isolation**: Tests must be independent and not rely on external state

### Performance Requirements
- **Model Loading**: Optimize for lazy loading; models should only load when needed
- **Response Time**: Triage assessment must complete in <10 seconds for normal inputs
- **Memory Management**: Proper cleanup of GPU resources; no memory leaks
- **Caching**: Reuse loaded models across multiple requests when possible

## Security Principles

### Credential Management
- **No Hardcoded Secrets**: Zero tolerance for hardcoded API keys, tokens, or passwords
- **Environment Variables**: All sensitive data must come from environment variables
- **Token Validation**: HF_TOKEN must be validated before model loading attempts
- **Logging**: Never log sensitive information (tokens, patient data in production)

### Data Privacy
- **Patient Data**: Treat all patient scenarios as potentially sensitive
- **Minimal Storage**: Do not persist task inputs or outputs beyond execution
- **Compliance**: Design with HIPAA/GDPR principles in mind (though this is educational software)

## Compliance & Standardization

### A2A Protocol Adherence
- **Strict Schema Compliance**: Input/output schemas must match `agent-card.json` exactly
- **Lifecycle Methods**: Implement all required A2A agent lifecycle methods (reset, health_check)
- **Error Responses**: Return structured error objects that conform to A2A error schema
- **Versioning**: Agent card version must be incremented for breaking changes

### Code Style
- **PEP 8**: Python code must follow PEP 8 style guidelines
- **Linting**: All code must pass `flake8` linting before PR approval
- **Formatting**: Use consistent formatting (recommend `black` for auto-formatting)

## Development Workflow

### Git Practices
- **Branch Protection**: Main branch requires PR review; no direct commits
- **Commit Messages**: Use conventional commits format (feat:, fix:, docs:, etc.)
- **Small PRs**: Keep pull requests focused and reviewable (<500 lines)

### CI/CD Integration
- **Pre-commit Checks**: Run linting and formatting checks locally before pushing
- **GitHub Actions**: All tests must pass in CI before merge
- **Dual-Mode Testing**: CI must validate both Gradio and A2A modes
- **Docker Builds**: Verify Docker images build successfully on every PR

## Technical Decisions Framework

When making implementation choices, prioritize in this order:

1. **User Safety**: Never compromise on medical accuracy or safety warnings
2. **Protocol Compliance**: A2A specification adherence is non-negotiable
3. **Performance**: Optimize for response time and resource efficiency
4. **Maintainability**: Choose readable, well-documented solutions over clever hacks
5. **Extensibility**: Design for future additions (e.g., new triage systems, multi-modal inputs)

## Educational Mission

### Transparency
- **Model Cards**: Maintain accurate, up-to-date model cards with training metrics
- **Limitations**: Clearly document model limitations and known failure modes
- **Reproducibility**: Provide complete instructions for reproducing training and deployment

### Accessibility
- **Open Source**: All code under permissive licenses (MIT/Apache 2.0 where possible)
- **Documentation**: Write docs for nursing students, not just ML engineers
- **Examples**: Provide diverse, realistic clinical scenarios in examples

## Review & Updates

This constitution should be reviewed quarterly and updated as the project evolves. All team members and contributors are expected to understand and uphold these principles.

---

*Last Updated: 2026-01-11*  
*Version: 1.0.0*
