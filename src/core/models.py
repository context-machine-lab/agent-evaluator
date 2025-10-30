"""Model manager for multi-model support based on DeepResearchAgent patterns."""

from typing import Any, Dict, Optional, List, AsyncGenerator, Union
from dataclasses import dataclass
from .registry import MODEL, register_model
from .memory import ChatMessage, TokenUsage, ToolCall

# Import claude_agent_sdk
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    retry_count: int = 3

class BaseModel:
    """Base class for language models."""

    def __init__(self, config: ModelConfig):
        """Initialize model with configuration.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model_id = config.model_id

    async def __call__(
        self,
        messages: List[ChatMessage],
        tools_to_call_from: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChatMessage:
        """Call the model with messages.

        Args:
            messages: Input messages
            tools_to_call_from: Available tools for function calling
            **kwargs: Additional parameters

        Returns:
            Model response as ChatMessage
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count estimate
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4


@register_model("claude_agent")
class ClaudeAgentModel(BaseModel):
    """Model wrapper for claude_agent_sdk."""

    def __init__(self, config: ModelConfig):
        """Initialize Claude Agent SDK model."""
        super().__init__(config)
        # Map model IDs to claude_agent_sdk model names
        self.model_map = {
            "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
        }

        self.model_name = self.model_map.get(config.model_id, config.model_id)

    async def __call__(
        self,
        messages: List[ChatMessage],
        tools_to_call_from: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatMessage, AsyncGenerator[Dict[str, Any], None]]:
        """Call claude_agent_sdk with messages.

        Args:
            messages: Input messages
            tools_to_call_from: Available SDK tools (dict with tool names as keys and values)
            stream: Whether to stream intermediate events
            **kwargs: Additional parameters

        Returns:
            Model response as ChatMessage or stream of events
        """
        if stream:
            return self._stream_call(messages, tools_to_call_from, **kwargs)
        else:
            # Non-streaming version - collect all events
            content = ""
            tool_calls = []

            async for event in self._stream_call(messages, tools_to_call_from, **kwargs):
                if event["type"] == "text":
                    content += event["text"]
                elif event["type"] == "tool_call":
                    tool_calls.append(event["tool_call"])
                elif event["type"] == "final":
                    content = event["content"]
                    if event.get("tool_calls"):
                        tool_calls = event["tool_calls"]

            return ChatMessage(
                role="assistant",
                content=content.strip() if content else "",
                tool_calls=tool_calls if tool_calls else None
            )

    async def _stream_call(
        self,
        messages: List[ChatMessage],
        tools_to_call_from: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream events from claude_agent_sdk.

        Yields:
            Dict with event type and data
        """
        # Convert messages to prompt string
        prompt = self._convert_messages_to_prompt(messages, tools_to_call_from)

        # Get SDK tool names directly
        allowed_tools = list(tools_to_call_from.keys()) if tools_to_call_from else []

        # Configure options - let SDK handle tool execution with multiple turns
        options = ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            model=self.model_name,
            max_turns=30,  # Allow SDK to use tools up to 30 times per agent step
            permission_mode="bypassPermissions",  # For automated tasks
        )

        # Call claude_agent_sdk - it will handle tool execution internally
        try:
            content = ""
            tool_calls = []

            # Execute query - SDK handles tool execution automatically
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    # Process content blocks
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            content += block.text
                            # Yield text event
                            yield {"type": "text", "text": block.text}
                        elif isinstance(block, ToolUseBlock):
                            # SDK is making a tool call
                            tool_call = ToolCall(
                                id=block.id,
                                name=block.name,
                                arguments=block.input if isinstance(block.input, dict) else {"input": block.input}
                            )
                            tool_calls.append(tool_call)
                            # Yield tool call event
                            yield {"type": "tool_call", "tool_call": tool_call}
                elif isinstance(message, ResultMessage):
                    # Final result from SDK
                    if message.result:
                        content += f"\n{message.result}"
                        yield {"type": "result", "result": message.result}

            # Yield final event with complete content
            yield {
                "type": "final",
                "content": content.strip() if content else "",
                "tool_calls": tool_calls if tool_calls else None
            }

        except Exception as e:
            # Yield error event
            yield {
                "type": "error",
                "error": f"Error calling claude_agent_sdk: {str(e)}"
            }

    def _convert_messages_to_prompt(self, messages: List[ChatMessage], tools: Optional[Dict[str, Any]] = None) -> str:
        """Convert internal messages to a prompt string for claude_agent_sdk.

        Args:
            messages: List of chat messages
            tools: Available tools (not used - SDK handles tool context)

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                # Add system prompts at the beginning
                prompt_parts.insert(0, f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                # Include tool results
                prompt_parts.append(f"Tool Result: {msg.content}")

        # Join with double newlines for clarity
        prompt = "\n\n".join(prompt_parts)

        # If the last message wasn't from user, add a continuation prompt
        if messages and messages[-1].role != "user":
            prompt += "\n\nHuman: Please continue."

        return prompt