"""Base async multi-step agent implementation based on DeepResearchAgent patterns."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict, AsyncIterator
from dataclasses import dataclass, field
import time
from datetime import datetime

# Import types from memory (to be implemented)
from ..core.memory import (
    AgentMemory,
    ActionStep,
    TaskStep,
    PlanningStep,
    FinalAnswerStep,
    ChatMessage,
    ToolCall,
    TokenUsage,
    AgentError,
)


class PlanningPromptTemplate(TypedDict):
    """Template for planning prompts."""
    prompt: str


class ManagedAgentPromptTemplate(TypedDict):
    """Template for managed agent communication."""
    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """Template for final answer formatting."""
    prompt: str
    format: str


class PromptTemplates(TypedDict):
    """Complete prompt template structure for an agent."""
    system_prompt: str
    user_prompt: str
    task_instruction: str
    planning: Optional[PlanningPromptTemplate]
    managed_agent: Optional[ManagedAgentPromptTemplate]
    final_answer: Optional[FinalAnswerPromptTemplate]


@dataclass
class StepOutput:
    """Output from a single step execution."""
    output: Any
    is_final_answer: bool = False
    error: Optional[AgentError] = None
    observations: Optional[str] = None
    observations_images: Optional[List[Any]] = None


class AsyncMultiStepAgent(ABC):
    """Base class for async multi-step agents with memory and tool execution."""

    def __init__(
        self,
        name: str,
        model: Any,  # Model instance (can be claude_agent_sdk model or custom)
        tools: Optional[Dict[str, Any]] = None,
        managed_agents: Optional[List["AsyncMultiStepAgent"]] = None,
        max_steps: int = 10,
        planning_interval: Optional[int] = None,
        provide_run_summary: bool = False,
        prompt_templates: Optional[PromptTemplates] = None,
        **kwargs
    ):
        """Initialize async multi-step agent.

        Args:
            name: Agent name
            model: LLM model instance
            tools: Dictionary of available tools
            managed_agents: List of child agents (for hierarchical agents)
            max_steps: Maximum execution steps
            planning_interval: Steps between planning phases (None = no planning)
            provide_run_summary: Include execution summary in output
            prompt_templates: Agent-specific prompts
        """
        self.name = name
        self.model = model
        self.tools = tools or {}
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self.provide_run_summary = provide_run_summary
        self.prompt_templates = prompt_templates or self._default_prompt_templates()

        # Setup managed agents (child agents as tools)
        self._setup_managed_agents(managed_agents)

        # Combine tools and managed agents
        self.tools_and_managed_agents = {**self.tools, **self.managed_agents}

        # Initialize memory
        self.memory: Optional[AgentMemory] = None
        self.task: Optional[str] = None
        self.step_number: int = 1

        # For managed agent protocol
        self.inputs = None
        self.output_type = None

    def _default_prompt_templates(self) -> PromptTemplates:
        """Provide default prompt templates."""
        return PromptTemplates(
            system_prompt="You are a helpful assistant.",
            user_prompt="",
            task_instruction="Complete the following task:\n{{task}}",
            planning=None,
            managed_agent=None,
            final_answer=None
        )

    def _setup_managed_agents(self, managed_agents: Optional[List["AsyncMultiStepAgent"]] = None):
        """Setup managed agents as callable tools.

        This allows child agents to be invoked like tools by the parent agent.
        """
        self.managed_agents = {}
        if managed_agents:
            for agent in managed_agents:
                self.managed_agents[agent.name] = agent
                # Set inputs/outputs for tool-like interface
                agent.inputs = {
                    "task": {"type": "string", "description": "Long detailed description of the task."},
                    "additional_args": {"type": "object", "description": "Dictionary of extra inputs."}
                }
                agent.output_type = "string"

    def initialize_system_prompt(self) -> str:
        """Initialize system prompt with task-specific instructions."""
        system_prompt = self.prompt_templates["system_prompt"]
        if "task_instruction" in self.prompt_templates:
            task_instruction = self.prompt_templates["task_instruction"]
            # Replace {{task}} with actual task
            if self.task:
                task_instruction = task_instruction.replace("{{task}}", self.task)
            system_prompt = f"{system_prompt}\n\n{task_instruction}"
        return system_prompt

    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        max_steps: Optional[int] = None,
        images: Optional[List[Any]] = None,
        **kwargs
    ) -> Any:
        """Run agent on a task.

        Args:
            task: Task description
            stream: Enable streaming mode
            reset: Reset memory before running
            max_steps: Override default max steps
            images: Optional images for the task

        Returns:
            Final answer from agent
        """
        if max_steps is None:
            max_steps = self.max_steps

        # Initialize task and memory
        self.task = task
        self.system_prompt = self.initialize_system_prompt()

        if reset or self.memory is None:
            user_prompt = self.prompt_templates.get("user_prompt", "")
            self.memory = AgentMemory(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt if user_prompt else None
            )
            self.step_number = 1

        # Add task to memory
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        # Run agent steps
        if stream:
            # Return async generator for streaming
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        else:
            # Collect all steps and return final answer
            steps = []
            async for step in self._run_stream(task=self.task, max_steps=max_steps, images=images):
                steps.append(step)

            # Extract final answer
            if steps and isinstance(steps[-1], FinalAnswerStep):
                return steps[-1].output
            else:
                return await self._handle_max_steps_reached(task, images)

    async def _run_stream(
        self,
        task: str,
        max_steps: int,
        images: Optional[List[Any]] = None
    ) -> AsyncIterator[Any]:
        """Stream agent execution steps.

        Yields:
            Step outputs including intermediate results and final answer
        """
        returned_final_answer = False
        final_answer = None

        while not returned_final_answer and self.step_number <= max_steps:
            # Optional planning step
            if self.planning_interval is not None and self.step_number % self.planning_interval == 0:
                planning_step = await self._generate_planning_step(task, images)
                if planning_step:
                    self.memory.steps.append(planning_step)
                    yield planning_step

            # Action step
            action_step = ActionStep(
                step_number=self.step_number,
                timing={"start": time.time()},
                model_input_messages=None,
                tool_calls=None,
                error=None,
                model_output_message=None,
                model_output=None,
                observations=None,
                observations_images=None,
                action_output=None,
                token_usage=None,
                is_final_answer=False
            )

            # Add to memory immediately (for real-time monitoring)
            self.memory.steps.append(action_step)

            # Execute step
            async for output in self._step_stream(action_step):
                if output.is_final_answer:
                    returned_final_answer = True
                    final_answer = output.output
                    action_step.is_final_answer = True
                    action_step.action_output = final_answer

                # Update action step with output (updates memory in real-time)
                if output.observations:
                    action_step.observations = output.observations
                if output.observations_images:
                    action_step.observations_images = output.observations_images
                if output.error:
                    action_step.error = output.error

                yield output

            # Record timing
            action_step.timing["end"] = time.time()
            self.step_number += 1

        # Handle max steps reached
        if not returned_final_answer and self.step_number == max_steps + 1:
            final_answer = await self._handle_max_steps_reached(task, images)
            final_step = FinalAnswerStep(output=final_answer)
            self.memory.steps.append(final_step)
            yield final_step

    @abstractmethod
    async def _step_stream(self, action_step: ActionStep) -> AsyncIterator[StepOutput]:
        """Execute a single step and stream outputs.

        This must be implemented by subclasses to define agent behavior.

        Args:
            action_step: Step to execute

        Yields:
            Step outputs
        """
        pass

    async def _generate_planning_step(
        self,
        task: str,
        images: Optional[List[Any]] = None
    ) -> Optional[PlanningStep]:
        """Generate a planning step.

        Args:
            task: Current task
            images: Optional images

        Returns:
            Planning step or None
        """
        if not self.prompt_templates.get("planning"):
            return None

        planning_prompt = self.prompt_templates["planning"]["prompt"]
        # Generate plan using model
        # This is a simplified version - actual implementation would use the model
        plan = f"Planning for task: {task}"

        return PlanningStep(
            step_number=self.step_number,
            plan=plan,
            timing={"start": time.time(), "end": time.time()}
        )

    async def _handle_max_steps_reached(
        self,
        task: str,
        images: Optional[List[Any]] = None
    ) -> str:
        """Handle case when max steps is reached without final answer.

        Args:
            task: Original task
            images: Optional images

        Returns:
            Default final answer
        """
        return "Agent stopped due to iteration limit or time limit."

    async def write_memory_to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert memory to chat messages.

        Args:
            summary_mode: Use summary mode for context compression

        Returns:
            List of chat messages
        """
        if not self.memory:
            return []

        messages = []

        # Add system prompt
        if not summary_mode:
            messages.extend(self.memory.system_prompt.to_messages(summary_mode=summary_mode))

        # Add steps
        for step in self.memory.steps:
            messages.extend(step.to_messages(summary_mode=summary_mode))

        # Add user prompt
        if self.memory.user_prompt and not summary_mode:
            messages.extend(self.memory.user_prompt.to_messages(summary_mode=summary_mode))

        return messages


    async def __call__(self, task: str, **kwargs) -> str:
        """Make agent callable (for managed agent protocol).

        This is called when this agent is invoked as a managed agent by a parent.

        Args:
            task: Task from parent agent
            **kwargs: Additional arguments

        Returns:
            Formatted response with optional work summary
        """
        # Add managed agent task wrapping if template exists
        if self.prompt_templates.get("managed_agent"):
            task_template = self.prompt_templates["managed_agent"]["task"]
            # Simple template replacement (actual implementation would use Jinja2)
            full_task = task_template.replace("{{name}}", self.name).replace("{{task}}", task)
        else:
            full_task = task

        # Run the agent
        result = await self.run(full_task, **kwargs)

        # Format response
        if self.prompt_templates.get("managed_agent") and self.prompt_templates["managed_agent"].get("report"):
            report_template = self.prompt_templates["managed_agent"]["report"]
            answer = report_template.replace("{{name}}", self.name).replace("{{final_answer}}", str(result))
        else:
            answer = f"Agent {self.name} result:\n{result}"

        # Add work summary if requested
        if self.provide_run_summary:
            answer += "\n\n--- Work Summary ---\n"
            messages = await self.write_memory_to_messages(summary_mode=True)
            for message in messages:
                # Truncate long content
                content = str(message)[:1000]
                answer += f"{content}\n---\n"

        return answer