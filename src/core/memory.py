"""Memory system for agents based on DeepResearchAgent patterns."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import json
import time


@dataclass
class ChatMessage:
    """Represents a message in chat format."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    def __str__(self) -> str:
        """String representation."""
        if self.role == "tool":
            return f"Tool Result ({self.name}): {self.content}"
        elif self.tool_calls:
            calls = ", ".join([tc.name for tc in self.tool_calls])
            return f"{self.role.title()}: [Tool Calls: {calls}]"
        else:
            return f"{self.role.title()}: {self.content}"


@dataclass
class ToolCall:
    """Represents a tool call."""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": str(self.result) if self.result is not None else None
        }

    def __str__(self) -> str:
        """String representation."""
        args_str = json.dumps(self.arguments, indent=2) if self.arguments else "{}"
        return f"ToolCall({self.name}, args={args_str})"


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two token usage objects."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


@dataclass
class AgentError:
    """Represents an error during agent execution."""
    error_type: str
    message: str
    traceback: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        return f"{self.error_type}: {self.message}"


@dataclass
class Timing:
    """Timing information for a step."""
    start: float
    end: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate duration."""
        if self.end is not None:
            return self.end - self.start
        return None


class MemoryStep(ABC):
    """Base class for memory steps."""

    @abstractmethod
    def to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert step to chat messages.

        Args:
            summary_mode: Use summary mode for compression

        Returns:
            List of chat messages
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation."""
        pass


@dataclass
class ActionStep(MemoryStep):
    """Represents an action step in agent execution."""
    step_number: int
    timing: Union[Timing, Dict[str, float]]
    model_input_messages: Optional[List[ChatMessage]] = None
    tool_calls: Optional[List[ToolCall]] = None
    error: Optional[AgentError] = None
    model_output_message: Optional[ChatMessage] = None
    model_output: Optional[str] = None
    observations: Optional[str] = None
    observations_images: Optional[List[Any]] = None
    action_output: Any = None
    token_usage: Optional[TokenUsage] = None
    is_final_answer: bool = False

    def to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert to chat messages."""
        messages = []

        # In summary mode, skip model output text to save tokens
        if not summary_mode and self.model_output:
            messages.append(ChatMessage(
                role="assistant",
                content=self.model_output,
                tool_calls=self.tool_calls
            ))
        elif self.tool_calls:
            # Include tool calls even in summary mode
            messages.append(ChatMessage(
                role="assistant",
                content="",
                tool_calls=self.tool_calls
            ))

        # Include observations (tool results)
        if self.observations:
            for tool_call in (self.tool_calls or []):
                messages.append(ChatMessage(
                    role="tool",
                    content=self.observations,
                    tool_call_id=tool_call.id,
                    name=tool_call.name
                ))

        return messages

    def __str__(self) -> str:
        """String representation."""
        parts = [f"Step {self.step_number}"]

        if self.tool_calls:
            tools = ", ".join([tc.name for tc in self.tool_calls])
            parts.append(f"Tools: {tools}")

        if self.is_final_answer:
            parts.append("FINAL ANSWER")

        if self.error:
            parts.append(f"Error: {self.error}")

        if self.observations:
            obs = self.observations[:100] + "..." if len(self.observations) > 100 else self.observations
            parts.append(f"Observations: {obs}")

        if isinstance(self.timing, dict):
            duration = self.timing.get("end", 0) - self.timing.get("start", 0)
            if duration > 0:
                parts.append(f"Duration: {duration:.2f}s")

        return " | ".join(parts)


@dataclass
class PlanningStep(MemoryStep):
    """Represents a planning step."""
    step_number: int
    plan: str
    timing: Union[Timing, Dict[str, float]]

    def to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert to chat messages."""
        if summary_mode:
            # Skip planning in summary mode
            return []
        return [ChatMessage(role="assistant", content=f"[PLAN]\n{self.plan}")]

    def __str__(self) -> str:
        """String representation."""
        plan_preview = self.plan[:100] + "..." if len(self.plan) > 100 else self.plan
        return f"Planning Step {self.step_number}: {plan_preview}"


@dataclass
class TaskStep(MemoryStep):
    """Represents the user's task."""
    task: str
    task_images: Optional[List[Any]] = None

    def to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert to chat messages."""
        # Always include task
        content = self.task
        if self.task_images:
            content += f"\n[{len(self.task_images)} image(s) attached]"
        return [ChatMessage(role="user", content=content)]

    def __str__(self) -> str:
        """String representation."""
        task_preview = self.task[:200] + "..." if len(self.task) > 200 else self.task
        return f"Task: {task_preview}"


@dataclass
class SystemPromptStep(MemoryStep):
    """Represents the system prompt."""
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert to chat messages."""
        if summary_mode:
            # Skip system prompt in summary mode
            return []
        return [ChatMessage(role="system", content=self.system_prompt)]

    def __str__(self) -> str:
        """String representation."""
        prompt_preview = self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt
        return f"System: {prompt_preview}"


@dataclass
class UserPromptStep(MemoryStep):
    """Represents user-level prompt/instructions."""
    user_prompt: Optional[str]

    def to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert to chat messages."""
        if summary_mode or not self.user_prompt:
            # Skip user prompt in summary mode
            return []
        return [ChatMessage(role="user", content=self.user_prompt)]

    def __str__(self) -> str:
        """String representation."""
        if not self.user_prompt:
            return "User Prompt: (empty)"
        prompt_preview = self.user_prompt[:100] + "..." if len(self.user_prompt) > 100 else self.user_prompt
        return f"User Prompt: {prompt_preview}"


@dataclass
class FinalAnswerStep(MemoryStep):
    """Represents the final answer."""
    output: Any

    def to_messages(self, summary_mode: bool = False) -> List[ChatMessage]:
        """Convert to chat messages."""
        return [ChatMessage(role="assistant", content=str(self.output))]

    def __str__(self) -> str:
        """String representation."""
        output_str = str(self.output)
        output_preview = output_str[:200] + "..." if len(output_str) > 200 else output_str
        return f"Final Answer: {output_preview}"


@dataclass
class AgentMemory:
    """Agent's memory containing all execution steps."""
    system_prompt: SystemPromptStep
    user_prompt: UserPromptStep
    steps: List[Union[TaskStep, ActionStep, PlanningStep, FinalAnswerStep]] = field(default_factory=list)

    def __init__(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None
    ):
        """Initialize agent memory.

        Args:
            system_prompt: System prompt for the agent
            user_prompt: Optional user-level instructions
        """
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.user_prompt = UserPromptStep(user_prompt=user_prompt)
        self.steps = []

    def reset(self):
        """Reset memory steps."""
        self.steps = []

    def get_total_tokens(self) -> TokenUsage:
        """Calculate total token usage."""
        total = TokenUsage()
        for step in self.steps:
            if isinstance(step, ActionStep) and step.token_usage:
                total = total + step.token_usage
        return total

    def get_final_answer(self) -> Optional[Any]:
        """Get final answer if available."""
        for step in reversed(self.steps):
            if isinstance(step, FinalAnswerStep):
                return step.output
            if isinstance(step, ActionStep) and step.is_final_answer:
                return step.action_output
        return None

    def get_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls."""
        tool_calls = []
        for step in self.steps:
            if isinstance(step, ActionStep) and step.tool_calls:
                tool_calls.extend(step.tool_calls)
        return tool_calls

    def get_errors(self) -> List[AgentError]:
        """Get all errors."""
        errors = []
        for step in self.steps:
            if isinstance(step, ActionStep) and step.error:
                errors.append(step.error)
        return errors

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "system_prompt": self.system_prompt.system_prompt,
            "user_prompt": self.user_prompt.user_prompt,
            "steps": [str(step) for step in self.steps],
            "total_tokens": self.get_total_tokens().__dict__,
            "final_answer": self.get_final_answer(),
            "num_tool_calls": len(self.get_tool_calls()),
            "num_errors": len(self.get_errors())
        }

    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=== Agent Memory ===",
            f"Steps: {len(self.steps)}",
            f"Tool Calls: {len(self.get_tool_calls())}",
            f"Errors: {len(self.get_errors())}",
            f"Tokens: {self.get_total_tokens().total_tokens}",
        ]

        if self.steps:
            lines.append("\n--- Execution Steps ---")
            for step in self.steps:
                lines.append(str(step))

        final_answer = self.get_final_answer()
        if final_answer:
            lines.append(f"\n--- Final Answer ---\n{final_answer}")

        return "\n".join(lines)