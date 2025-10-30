# Agent1 - Research & Data Science Pipeline

A clean, modular implementation using Claude Agent SDK for deep research, data science workflows, and benchmark evaluations.

## Features

- **Deep Research Pipeline**: Multi-phase research on any topic with web search
- **Data Science Workflows**: Exploratory analysis, statistical analysis, and ML modeling
- **GAIA Benchmark Evaluation**: Evaluate Claude agents on the GAIA dataset
- **Hydra Configuration**: Clean configuration management with YAML files
- **Rich Console Output**: Beautiful progress tracking and logging
- **Async Execution**: Efficient concurrent task processing

## Installation

```bash
cd agent1
uv venv
uv pip install -e .
```

## Quick Start

### Deep Research

Research a topic:
```bash
python examples/dr.py research.topic="Impact of AI on healthcare"
```

Different research depths:
```bash
python examples/dr.py research.topic="Climate change" research=quick
python examples/dr.py research.topic="Quantum computing" research=exhaustive
```

Save output to file:
```bash
python examples/dr.py research.topic="AI Ethics" research.output_file=report.md
```

### Data Science

Analyze data:
```bash
python examples/ds.py data_science.task="Analyze sales trends" data_science.data_path=sales.csv
```

Build a model:
```bash
python examples/ds.py data_science.modeling.task="Predict customer churn" data_science.data_path=customers.csv
```

### GAIA Benchmark Evaluation

Run GAIA evaluation on validation set:
```bash
python examples/run_gaia.py gaia.split=validation gaia.max_tasks=5
```

Full test set evaluation:
```bash
python examples/run_gaia.py gaia.split=test
```

## Project Structure

```
agent1/
├── examples/
│   ├── dr.py               # Deep research CLI
│   ├── ds.py               # Data science CLI
│   └── run_gaia.py         # GAIA evaluation script
├── src/
│   ├── configs/
│   │   ├── deep_research.yaml  # Research configuration
│   │   ├── data_scientist.yaml # Data science configuration
│   │   └── gaia.yaml           # GAIA benchmark configuration
│   ├── claude.py           # Claude agent executor
│   ├── pipelines.py        # Pipeline implementations
│   ├── logger.py           # Rich console logger
│   └── gaia_utils.py       # GAIA dataset utilities
└── data/
    └── GAIA/              # GAIA dataset (add manually)
```

## Configuration

All configurations use Hydra and are stored in `src/configs/`. Key options:

### Model Configuration
- `model.name`: Claude model to use (default: claude-sonnet-4-5-20250929)
- `model.temperature`: Sampling temperature
- `model.max_tokens`: Maximum tokens

### Research Configuration
- `research.topic`: Research topic (required)
- `research.depth`: quick, standard, comprehensive, exhaustive
- `research.output_file`: Optional output file path

### GAIA Configuration
- `gaia.split`: validation or test
- `gaia.max_tasks`: Maximum tasks to evaluate
- `gaia.batch_size`: Concurrent batch size
- `gaia.results_path`: Output JSONL path

## GAIA Benchmark

### Setup

1. Download GAIA dataset to `data/GAIA/`:
   - `2023_validation.json` - Validation set with ground truth
   - `2023_test.json` - Test set without ground truth

2. Run evaluation:
```bash
python examples/run_gaia.py gaia.split=validation
```

### Features

- **Smart Resume**: Automatically skips completed tasks
- **Batch Processing**: Concurrent execution with configurable batch size
- **Comprehensive Metrics**: Accuracy calculation and detailed reports
- **Error Recovery**: Graceful error handling with detailed logging
- **Result Persistence**: JSONL format with metadata and costs

### Output Format

Results saved in JSONL:
```json
{
  "task_id": "test_001",
  "question": "What is 2 + 2?",
  "prediction": "4",
  "true_answer": "4",
  "tools_used": ["WebSearch"],
  "num_turns": 3,
  "cost_usd": 0.002,
  "duration_ms": 5432
}
```

## Testing

Test GAIA setup:
```bash
python test_gaia_setup.py
```

## API Components

### AgentExecutor
Executes single agents with specified tools and configurations:
```python
from src.claude import create_agent_executor

executor = create_agent_executor()
result = await executor.execute_agent(
    prompt="Research quantum computing",
    agent_type="research",
    allowed_tools=["WebSearch", "WebFetch"]
)
```

### PipelineExecutor
Orchestrates multi-phase pipelines:
```python
from src.claude import create_pipeline_executor

pipeline = create_pipeline_executor()
result = await pipeline.execute_pipeline(
    phases=[...],
    initial_context="Topic: AI"
)
```

### Research & Data Science Pipelines
High-level interfaces for specific workflows:
```python
from src.pipelines import DeepResearchPipeline, DataSciencePipeline

# Research
research = DeepResearchPipeline()
result = await research.research("AI ethics", depth="comprehensive")

# Data science
ds = DataSciencePipeline()
result = await ds.analyze_data(data_path="data.csv", analysis_type="exploratory")
```

## Development

### Adding New Pipelines

1. Create configuration in `src/configs/`
2. Extend `BasePipeline` in `src/pipelines.py`
3. Add CLI script in `examples/`

### Custom Agent Types

Modify `allowed_tools` in agent configurations:
- Research: WebSearch, WebFetch, Read, Write
- Analysis: Read, Write, Bash, Grep, Glob
- Coding: Read, Write, Edit, Bash

## Troubleshooting

- **Import Errors**: Ensure dependencies installed with `uv pip install -e .`
- **API Errors**: Check Claude API key is set
- **Dataset Not Found**: Download GAIA dataset to `data/GAIA/`
- **Out of Memory**: Reduce `batch_size` in configuration

## License

MIT