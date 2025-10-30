#!/usr/bin/env python3
"""GAIA Benchmark Evaluation - Simplified Entry Point"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

# Import new simplified modules
from src.config.loader import load_config
from src.core.logger import ResearchLogger
from src.core.models import MODEL, ModelConfig
from src.gaia.dataset import GAIADataset, GAIAScorer
from src.gaia.evaluator import get_tasks_to_run, run_batch, append_result
from src.agents.agent import create_agent, prepare_response
from src.tools.default_tools import default_tools


async def main():
    """Main GAIA benchmark evaluation entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='GAIA Benchmark Evaluation')
    parser.add_argument('--config', default='./configs/config.yaml', help='Configuration file')
    parser.add_argument('--max-tasks', type=int, help='Maximum tasks to run')
    parser.add_argument('--batch-size', type=int, help='Batch size for concurrent execution')
    parser.add_argument('--split', choices=['validation', 'test'], help='Dataset split')
    parser.add_argument('--output', help='Output results file path')
    args = parser.parse_args()

    # Load configuration
    logger = ResearchLogger()
    logger.section("GAIA Benchmark Evaluation")
    
    config = load_config(args.config, args)
    logger.info(f"Configuration loaded from: {args.config}")

    # Build model from config
    logger.phase_start("Initializing Model")
    model_data = config.model.to_dict()
    # Map 'name' to 'model_id' if present
    if 'name' in model_data:
        model_data['model_id'] = model_data.pop('name')
    model_config = ModelConfig(**model_data)
    model = MODEL.build({"type": "claude_agent"}, config=model_config)
    logger.success(f"Model initialized: {model.model_id}")

    # Load dataset
    logger.phase_start("Loading Dataset")
    dataset = GAIADataset(
        data_path=config.dataset['path'],
        split=config.dataset['split']
    )
    tasks = get_tasks_to_run(config, dataset)
    logger.success(f"Loaded {len(tasks)} tasks from {config.dataset['split']} split")

    if not tasks:
        logger.warning("No tasks to run")
        return

    # Process tasks in batches
    logger.section("Running Evaluation")
    all_results = []
    batch_size = config.concurrency

    # Helper functions for agent creation
    async def create_agent_fn(cfg):
        return await create_agent(cfg, default_tools=default_tools, model=model)

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:min(i + batch_size, len(tasks))]
        batch_num = (i // batch_size) + 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size

        logger.phase_start(f"Batch {batch_num}/{total_batches} ({len(batch)} tasks)")

        # Run batch
        results = await run_batch(config, batch, logger, create_agent_fn, prepare_response, model)

        # Save results
        for result in results:
            append_result(result, config.save_path)
            all_results.append(result)

        logger.success(f"Batch {batch_num} completed")

    # Compute metrics if configured
    if config.dataset.get('with_ground_truth') and config.evaluation.get('compute_metrics'):
        logger.section("Computing Metrics")
        scorer = GAIAScorer()
        metrics = scorer.compute_metrics(all_results)
        scorer.print_summary(metrics)

        # Save summary
        if config.get('output', {}).get('save_summary', True):
            summary_path = Path(config.save_path).with_suffix('.summary.md')
            summary = f"""# GAIA Evaluation Summary

## Configuration
- Agent: {config.agent_config['name']}
- Split: {config.dataset['split']}
- Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Results
- Total: {metrics['total']}
- Correct: {metrics['correct']}
- Accuracy: {metrics['accuracy']:.2f}%
"""
            summary_path.write_text(summary)
            logger.file(f"Summary saved to: {summary_path}")

    logger.section("Evaluation Complete!")
    logger.success(f"Results saved to: {config.save_path}")


if __name__ == '__main__':
    # Register agents
    import src.agents
    asyncio.run(main())