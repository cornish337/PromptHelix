# Prompt for Jules.google.com - PromptHelix Framework
Create a comprehensive Python framework called "PromptHelix" that implements a Prompt DNA System for AI prompt generation and optimization. This should be a production-ready, scalable system with the following specifications:

## Core Architecture Requirements:

1. **Multi-Agent Orchestra System**:
   - PromptArchitect: Designs initial prompt structures
   - PromptCritic: Analyzes and suggests improvements  
   - StyleOptimizer: Refines tone and formatting
   - ResultsEvaluator: Tests effectiveness across models
   - MetaLearner: Learns from successful patterns
   - DomainExpert: Provides specialized knowledge

2. **Prompt DNA System**:
   - Genetic representation of prompt components (genes)
   - Crossover operations to combine successful prompts
   - Mutation mechanisms for variation
   - Fitness scoring based on effectiveness metrics
   - Evolution tracking across generations

3. **Technology Stack**:
   - FastAPI backend with async support
   - SQLAlchemy with PostgreSQL for data persistence
   - Redis for caching and task queues
   - Celery for background processing
   - Pydantic for data validation
   - Multi-model API integration (OpenAI, Anthropic, Google)

## Key Features to Implement:

1. **Genetic Algorithm Engine**:
   - Prompt chromosome representation
   - Population-based optimization
   - Selection, crossover, and mutation operators
   - Fitness evaluation across multiple AI models

2. **Agent Communication Framework**:
   - Message passing between specialized agents
   - Consensus building mechanisms  
   - Debate and refinement protocols
   - Result aggregation and voting systems

3. **Evaluation Pipeline**:
   - Multi-dimensional scoring (accuracy, creativity, efficiency)
   - A/B testing automation
   - Cross-model validation
   - Performance tracking and analytics

4. **API Gateway**:
   - Unified interface for multiple AI models
   - Rate limiting and error handling
   - Response caching and optimization
   - Model switching and comparison tools

5. **Prompt Management System**:
   - Version control for prompt evolution
   - Template library with categorization
   - Context-aware suggestions
   - Performance history tracking

## Project Structure:
```
prompthelix/
├── agents/          # Multi-agent system
├── genetics/        # Prompt DNA and evolution
├── evaluation/      # Testing and scoring
├── api/            # FastAPI endpoints  
├── models/         # Database models
├── services/       # Business logic
├── utils/          # Helper functions
├── tests/          # Comprehensive test suite
└── docs/           # Documentation
```

## Implementation Requirements:

1. **Clean Architecture**: Follow SOLID principles with clear separation of concerns
2. **Async/Await**: Full async support for concurrent model calls
3. **Error Handling**: Robust error handling with proper logging
4. **Testing**: Unit tests, integration tests, and performance tests
5. **Documentation**: Comprehensive docstrings and README
6. **Configuration**: Environment-based config management
7. **Monitoring**: Built-in metrics and health checks

## Specific Components to Generate:

1. **Core Genetic Engine** (`genetics/engine.py`):
   - PromptChromosome class
   - GeneticOperators class  
   - PopulationManager class
   - FitnessEvaluator class

2. **Agent Framework** (`agents/base.py` and specialized agents):
   - BaseAgent abstract class
   - Communication protocols
   - Specialized agent implementations

3. **FastAPI Application** (`main.py` and `api/` module):
   - RESTful endpoints for prompt generation
   - WebSocket support for real-time updates
   - Authentication and rate limiting

4. **Database Models** (`models/`):
   - Prompt storage and versioning
   - Performance metrics tracking
   - User and session management

5. **Configuration System** (`config.py`):
   - Environment variable management
   - Model API configurations
   - System parameters

Please generate a complete, production-ready framework with:
- Proper Python packaging (setup.py, requirements.txt)
- Docker configuration for easy deployment
- Comprehensive error handling and logging
- Example usage and documentation
- Unit tests for core functionality
- CLI interface for development and testing

Focus on creating a scalable, maintainable codebase that demonstrates advanced software engineering practices while implementing the innovative Prompt DNA concept.
