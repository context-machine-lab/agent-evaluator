"""Stateless pipeline for agent execution."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class ExecutionContext:
    """Stateful context for a single execution"""
    task_id: str
    task: str
    agent: Any  # The actual agent instance
    start_time: float
    memory: Dict[str, Any] = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = {}


class Pipeline:
    """Stateless pipeline that creates stateful contexts for execution."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary (immutable)
        """
        self.config = config
        self._task_counter = 0

    async def execute(self, task: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a task in an isolated context.

        Args:
            task: The task to execute
            context_id: Optional ID for the execution context

        Returns:
            Dict containing result, context_id, and metrics
        """
        # Create a fresh execution context for this task
        context = await self._create_context(task, context_id)

        try:
            # Run in isolated context
            result = await self._run_in_context(context)

            # Return result with optional context preservation
            return {
                'result': result,
                'context_id': context.task_id,
                'metrics': self._extract_metrics(context),
                'success': True
            }
        except Exception as e:
            return {
                'result': None,
                'context_id': context.task_id,
                'error': str(e),
                'success': False
            }
        finally:
            # Clean up context unless explicitly preserved
            if not self.config.get('preserve_contexts', False):
                await self._cleanup_context(context)

    async def execute_batch(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel, each in isolation.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of results from each task
        """
        return await asyncio.gather(*[
            self.execute(task) for task in tasks
        ])

    async def _create_context(self, task: str, context_id: Optional[str] = None) -> ExecutionContext:
        """Create a new execution context.

        Args:
            task: The task for this context
            context_id: Optional ID, will generate if not provided

        Returns:
            New ExecutionContext instance
        """
        # Generate ID if not provided
        if not context_id:
            self._task_counter += 1
            context_id = f"task_{self._task_counter}_{int(time.time())}"

        # Create agent for this context
        agent = await self._create_agent()

        return ExecutionContext(
            task_id=context_id,
            task=task,
            agent=agent,
            start_time=time.time()
        )

    async def _create_agent(self) -> Any:
        """Create a fresh agent instance for a context.

        Returns:
            Agent instance
        """
        # This will be implemented to use the agent factory
        # For now, returning None as placeholder
        from ..agents.factory import AgentFactory

        agent_type = self.config.get('agent', {}).get('type', 'general')
        agent_config = self.config.get('agent', {})

        return await AgentFactory.create(agent_type, **agent_config)

    async def _run_in_context(self, context: ExecutionContext) -> Any:
        """Run task in the given context.

        Args:
            context: The execution context

        Returns:
            Task result
        """
        # Execute the task with the agent
        result = await context.agent.run(task=context.task)

        # Store execution metadata
        context.memory['execution_time'] = time.time() - context.start_time
        context.memory['result'] = result

        return result

    def _extract_metrics(self, context: ExecutionContext) -> Dict[str, Any]:
        """Extract metrics from execution context.

        Args:
            context: The execution context

        Returns:
            Dict of metrics
        """
        metrics = {
            'execution_time': context.memory.get('execution_time', 0),
            'task_id': context.task_id,
        }

        # Add token usage if available
        if hasattr(context.agent, 'memory') and context.agent.memory:
            token_usage = context.agent.memory.get_total_tokens()
            if token_usage:
                metrics['tokens'] = {
                    'input': token_usage.input_tokens,
                    'output': token_usage.output_tokens,
                    'total': token_usage.total_tokens
                }

        return metrics

    async def _cleanup_context(self, context: ExecutionContext) -> None:
        """Clean up resources from execution context.

        Args:
            context: The execution context to clean up
        """
        # Clean up agent resources if needed
        if hasattr(context.agent, 'cleanup'):
            await context.agent.cleanup()

        # Clear memory
        context.memory.clear()