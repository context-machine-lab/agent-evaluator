"""GAIA benchmark evaluation logic."""

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .monitor import monitor_agent_progress

# Thread-safe lock for appending results
result_lock = threading.Lock()


def append_result(result: Dict[str, Any], results_path: str) -> None:
    """Append result to JSONL file (thread-safe)

    Args:
        result: Result dictionary to save
        results_path: Path to results file
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with result_lock:
        with open(results_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')


def get_tasks_to_run(config, dataset) -> List[dict]:
    """Get tasks to run, filtering out completed ones if configured.

    Args:
        config: Configuration object
        dataset: GAIA dataset

    Returns:
        List of tasks to run
    """
    all_tasks = dataset.get_tasks(max_tasks=getattr(config, 'max_tasks', None))

    # Filter completed tasks if configured
    if getattr(config, 'filter_completed', True):
        results_path = config.save_path
        if Path(results_path).exists():
            try:
                completed = []
                with open(results_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if 'task_id' in entry:
                                completed.append(entry['task_id'])
                        except json.JSONDecodeError:
                            continue

                tasks_to_run = [t for t in all_tasks if t['task_id'] not in completed]
                print(f"Filtering: {len(all_tasks)} total, {len(completed)} completed, {len(tasks_to_run)} to run")
            except Exception as e:
                print(f"Warning: Error filtering tasks: {e}")
                tasks_to_run = all_tasks
        else:
            tasks_to_run = all_tasks
    else:
        tasks_to_run = all_tasks

    return tasks_to_run


async def answer_single_question(config, example, logger, create_agent_fn, prepare_response_fn, model=None) -> Dict[str, Any]:
    """Execute a single GAIA task using the agent architecture.

    Args:
        config: Configuration object
        example: GAIA task dictionary
        logger: Logger instance
        create_agent_fn: Function to create agent
        prepare_response_fn: Function to prepare response
        model: Optional model instance for reformulation

    Returns:
        Result dictionary with prediction and metadata
    """
    task_id = example['task_id']
    question = example['question']
    true_answer = example.get('final_answer', '?')

    # Augment question with file information
    augmented_question = question
    if example.get('file_name'):
        augmented_question += f"\n\nTo solve this task, you will need to use this attached file: {example['file_name']}"

    # Record start time
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        logger.phase_start(f"Task {task_id}", phase_num=None, total_phases=None)
        logger.info(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")
        if true_answer != "?":
            logger.info(f"Expected: {true_answer}")

        # Create agent
        agent = await create_agent_fn(config)
        logger.info(f"Using agent: {agent.name}")
        logger.print_raw("")  # Add blank line before progress logs

        # Create stop event for monitoring
        stop_event = asyncio.Event()

        # Start monitoring task to show progress in real-time
        monitor_task = asyncio.create_task(
            monitor_agent_progress(agent, logger, task_id, stop_event)
        )

        # Run agent (monitoring will show progress concurrently)
        final_result = await agent.run(task=augmented_question)

        # Stop monitoring
        stop_event.set()
        await monitor_task

        # Get agent memory for response preparation
        agent_memory = await agent.write_memory_to_messages(summary_mode=True)

        # Prepare final response with reformulation if configured
        if getattr(config.response_preparation, 'use_reformulation', False):
            # If model_manager not provided, create a default one
            if model_manager is None:
                from ..core.models import ModelManager
                model_manager = ModelManager.create(
                    getattr(config, 'model_configs', None),
                    getattr(config, 'use_local_proxy', False)
                )
            reformulation_model_id = config.response_preparation.get('reformulation_model', 'gemini-2.5-flash')
            reformulation_model = model_manager.get_model(reformulation_model_id)
            final_result = await prepare_response_fn(
                augmented_question,
                agent_memory,
                reformulation_model=reformulation_model
            )

        # Extract clean answer using AnswerProcessor
        from .answer_processor import AnswerProcessor
        processed_response = AnswerProcessor.process_agent_response(str(final_result))
        prediction = processed_response["answer"]
        raw_prediction = str(final_result)  # Keep raw for debugging

        # Extract intermediate steps
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False

        if agent.memory:
            for step in agent.memory.steps:
                step_str = str(step)
                intermediate_steps.append(step_str)

                # Check for errors
                if "AgentParsingError" in step_str:
                    parsing_error = True
                if hasattr(step, 'error') and step.error:
                    parsing_error = True

            # Check if iteration limit was reached
            if "Agent stopped due to iteration limit" in prediction:
                iteration_limit_exceeded = True

            # Get token usage
            token_usage = agent.memory.get_total_tokens()
        else:
            token_usage = None

        # Build result
        result = {
            'agent_name': config.agent_config['name'],
            'task_id': task_id,
            'question': question,
            'augmented_question': augmented_question,
            'prediction': prediction,
            'raw_prediction': raw_prediction,  # Include raw for debugging
            'extraction_metadata': processed_response["metadata"],  # Include extraction metadata
            'true_answer': true_answer,
            'intermediate_steps': intermediate_steps,
            'parsing_error': parsing_error,
            'iteration_limit_exceeded': iteration_limit_exceeded,
            'agent_error': None,
            'start_time': start_time,
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'token_usage': token_usage.__dict__ if token_usage else None,
        }

        logger.success(f"Task {task_id} completed: {prediction[:100]}...")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        result = {
            'agent_name': config.agent_config.get('name', 'unknown'),
            'task_id': task_id,
            'question': question,
            'augmented_question': augmented_question,
            'prediction': None,
            'true_answer': true_answer,
            'intermediate_steps': [],
            'parsing_error': False,
            'iteration_limit_exceeded': False,
            'agent_error': str(e),
            'start_time': start_time,
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    return result


async def run_batch(
    config,
    tasks: List[Dict[str, Any]],
    logger,
    create_agent_fn,
    prepare_response_fn,
    model_manager=None
) -> List[Dict[str, Any]]:
    """Run a batch of tasks concurrently.

    Args:
        config: Configuration object
        tasks: List of tasks to run
        logger: Logger instance
        create_agent_fn: Function to create agent
        prepare_response_fn: Function to prepare response
        model_manager: Optional ModelManager instance for model access

    Returns:
        List of results
    """
    # Create coroutines for concurrent execution
    coroutines = [
        answer_single_question(config, task, logger, create_agent_fn, prepare_response_fn, model_manager)
        for task in tasks
    ]

    # Execute concurrently
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            processed_results.append({
                'task_id': tasks[i]['task_id'],
                'prediction': None,
                'agent_error': str(result),
                'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            processed_results.append(result)

    return processed_results