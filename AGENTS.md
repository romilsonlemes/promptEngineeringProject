# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Folder-Specific Setup and Dependencies

### Chapter 1: Tipos de Prompts
```bash
cd 1-tipos-de-prompts/

# Setup
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=...
```

**Key Dependencies:**
- langchain==0.3.27
- langchain-openai==0.3.32
- openai==1.102.0
- rich==14.1.0
- python-dotenv==1.1.1

**Scripts:**
```bash
python 0-Role-prompting.py          # Role-based prompting
python 1-zero-shot.py               # Zero-shot prompting
python 2-one-few-shot.py            # One-shot and few-shot examples
python 3-CoT.py                     # Chain of Thought
python 3.1-CoT-Self-consistency.py  # CoT with self-consistency
python 4-ToT.py                     # Tree of Thoughts
python 5-SoT.py                     # Skeleton of Thought
python 6-ReAct.py                   # ReAct framework
python 7-Prompt-channing.py         # Prompt chaining (generates output file)
python 8-Least-to-most.py           # Least-to-most decomposition
```

### Chapter 4: Prompts e Workflow de Agentes
```bash
cd 4-prompts-e-workflow-de-agentes/

# No specific requirements.txt - uses root dependencies
# Structure:
#   agents/    - Agent implementations for architectural analysis
#   commands/  - Command implementations for agent orchestration
```

### Chapter 5: Gerenciamento e Versionamento de Prompts
```bash
cd 5-gerenciamento-e-versionamento-de-prompts/

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

**Environment Variables:**
```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional (for LangSmith integration)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-key-here
LANGCHAIN_PROJECT=prompt-management-system
```

**Key Dependencies (updated versions):**
- langchain==1.0.0a5
- langchain-core==0.3.76
- langchain-openai==0.3.33
- langgraph==0.6.7
- langgraph-prebuilt==0.6.4
- langsmith==0.4.29
- pytest==8.3.4
- Jinja2==3.1.6

**Commands:**
```bash
# Run agents
python src/agent_code_reviewer.py
python src/agent_pull_request.py

# Run tests
pytest tests/test_prompts.py -v
pytest tests/ -v  # All tests
```

### Chapter 6: Prompt Enriquecido
```bash
cd 6-prompt-enriquecido/

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=KEY
```

**Key Dependencies (same as Chapter 5):**
- langchain==1.0.0a5
- langchain-core==0.3.76
- langchain-openai==0.3.33
- langgraph==0.6.7
- langsmith==0.4.29
- openai==1.108.0
- pytest==8.3.4

**Scripts:**
```bash
python 0-No-expansion.py          # Basic prompting without expansion
python 1-ITER_RETGEN.py           # Iterative retrieval generation
python 2-query-enrichment.py      # Query enrichment techniques
```

**Additional Resource:**
- `repo_langchain_1.0/`: Contains LangChain reference implementation for testing

## Project Architecture

### Core Structure
- **1-tipos-de-prompts/**: Fundamental prompt engineering techniques
  - 9 example scripts demonstrating various prompting strategies
  - `utils.py`: Shared utility for Rich-formatted console output
  - Own `requirements.txt` and `.env` configuration

- **4-prompts-e-workflow-de-agentes/**: Agent-based workflow implementations
  - `agents/`: Specialized agents for code analysis
  - `commands/`: Command layer for agent coordination
  - Uses root project dependencies

- **5-gerenciamento-e-versionamento-de-prompts/**: Advanced prompt management
  - `src/`: Agent implementations for code review and PR creation
  - `tests/`: Pytest-based prompt validation
  - `prompts/`: Versioned prompt storage with registry system
  - Supports both local and LangSmith remote management
  - Own `requirements.txt` with newer LangChain versions and LangGraph

- **6-prompt-enriquecido/**: Advanced prompt enrichment techniques
  - Query expansion and enrichment examples
  - ITER-RETGEN (Iterative Retrieval Generation) implementation
  - Uses same dependencies as Chapter 5
  - Includes LangChain repository for reference

### Common Patterns Across All Folders
- All use `python-dotenv` for environment configuration
- LangChain framework as primary LLM interaction layer
- Rich library for enhanced terminal output (Chapter 1)
- Model flexibility with commented alternatives
- Consistent error handling with descriptive messages

### Version Differences Between Folders

**Chapter 1 (Basic examples):**
- Uses stable LangChain 0.3.27
- Basic dependencies for prompt engineering demos

**Chapters 5 & 6 (Advanced features):**
- Uses LangChain 1.0.0a5 (alpha version)
- Includes LangGraph for graph-based workflows
- Adds Jinja2 for template processing
- Includes pytest for testing

## Testing Commands
```bash
# Chapter 5 & 6 (with pytest)
cd 5-gerenciamento-e-versionamento-de-prompts/
pytest tests/ -v                    # All tests
pytest tests/test_prompts.py -v     # Specific test file
pytest -k "test_name" -v            # Test by name pattern
```

## Output Files
- **Chapter 1:** `prompt_chaining_result.md` - Generated by 7-Prompt-channing.py
- **Chapter 5:** Various `.json` and `.yaml` files for prompt versioning in `prompts/` directory

## Quick Start for Each Chapter
```bash
# Chapter 1 - Basic Prompt Engineering
cd 1-tipos-de-prompts && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
# Add OPENAI_API_KEY to .env
python 1-zero-shot.py

# Chapter 5 - Prompt Management
cd 5-gerenciamento-e-versionamento-de-prompts && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
# Add OPENAI_API_KEY and optionally LangSmith keys to .env
python src/agent_code_reviewer.py

# Chapter 6 - Enriched Prompts
cd 6-prompt-enriquecido && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
# Add OPENAI_API_KEY to .env
python 2-query-enrichment.py
```