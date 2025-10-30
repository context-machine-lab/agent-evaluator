"""General agent implementation with tool calling based on DeepResearchAgent patterns."""

import asyncio
import json
import time
import traceback
from typing import Any, Dict, List, Optional, AsyncIterator

from .base import AsyncMultiStepAgent, StepOutput
from ..core.memory import (
    ActionStep,
    ChatMessage,
    ToolCall,
    TokenUsage,
    AgentError,
)
from ..core.registry import register_agent


@register_agent()
class GeneralAgent(AsyncMultiStepAgent):
    """General purpose agent with tool calling capabilities."""

    def __init__(
        self,
        name: str = "general_agent",
        model: Any = None,
        tools: Optional[Dict[str, Any]] = None,
        managed_agents: Optional[List[AsyncMultiStepAgent]] = None,
        max_steps: int = 10,
        planning_interval: Optional[int] = None,
        provide_run_summary: bool = False,
        prompt_templates: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize general agent.

        Args:
            name: Agent name
            model: Language model instance
            tools: Dictionary of available tools
            managed_agents: List of child agents
            max_steps: Maximum execution steps
            planning_interval: Steps between planning
            provide_run_summary: Include execution summary
            prompt_templates: Custom prompt templates
        """
        super().__init__(
            name=name,
            model=model,
            tools=tools,
            managed_agents=managed_agents,
            max_steps=max_steps,
            planning_interval=planning_interval,
            provide_run_summary=provide_run_summary,
            prompt_templates=prompt_templates,
            **kwargs
        )

    async def _step_stream(self, action_step: ActionStep) -> AsyncIterator[StepOutput]:
        """Execute a single step with tool calling.

        Args:
            action_step: Step to execute

        Yields:
            Step outputs including observations and final answers
        """
        try:
            # Prepare input messages for the model
            input_messages = await self.write_memory_to_messages(summary_mode=False)
            action_step.model_input_messages = input_messages

            # Track for incremental updates
            accumulated_content = ""
            tool_calls = []
            observations = []
            start_time = time.time()

            # Stream model response with tool execution
            stream = await self.model(
                input_messages,
                tools_to_call_from=self.tools_and_managed_agents if self.tools_and_managed_agents else None,
                stream=True  # Enable streaming
            )

            # Process streaming events
            async for event in stream:
                event_type = event.get("type")

                if event_type == "text":
                    # Model is generating text
                    text = event.get("text", "")
                    accumulated_content += text
                    action_step.model_output = accumulated_content

                elif event_type == "tool_call":
                    # Model is calling a tool
                    tool_call = event.get("tool_call")
                    if tool_call:
                        tool_calls.append(tool_call)
                        action_step.tool_calls = tool_calls

                        # Add observation placeholder
                        observations.append({
                            "tool_name": tool_call.name,
                            "tool_args": tool_call.arguments,
                            "result": "Executing..."
                        })
                        action_step.observations = observations

                elif event_type == "result":
                    # Tool execution result
                    result = event.get("result", "")
                    if observations and observations[-1]["result"] == "Executing...":
                        observations[-1]["result"] = result
                        action_step.observations = observations

                elif event_type == "final":
                    # Final complete response
                    final_content = event.get("content", "")
                    final_tool_calls = event.get("tool_calls")

                    if final_content:
                        action_step.model_output = final_content
                    if final_tool_calls:
                        action_step.tool_calls = final_tool_calls

                    # Estimate token usage
                    input_tokens = sum(len(msg.content) // 4 for msg in input_messages)
                    output_tokens = len(final_content) // 4 if final_content else 0
                    action_step.token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens
                    )

                    # Check if this is a final answer
                    is_final = self._is_final_answer(final_content, final_tool_calls)

                    if is_final:
                        yield StepOutput(
                            output=final_content,
                            is_final_answer=True
                        )
                    else:
                        yield StepOutput(
                            output=final_content,
                            observations=observations if observations else None,
                            is_final_answer=False
                        )

                elif event_type == "error":
                    # Handle errors from the SDK
                    error_msg = event.get("error", "Unknown error")
                    action_step.error = AgentError(
                        error_type="SDKError",
                        message=error_msg,
                        traceback=""
                    )
                    yield StepOutput(
                        output=None,
                        error=action_step.error,
                        is_final_answer=False
                    )

        except Exception as e:
            # Record error with full traceback
            action_step.error = AgentError(
                error_type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc()  # Capture full traceback for debugging
            )

            yield StepOutput(
                output=None,
                error=action_step.error,
                is_final_answer=False
            )

    async def process_tool_calls(
        self,
        chat_message: ChatMessage,
        action_step: ActionStep
    ) -> AsyncIterator[StepOutput]:
        """Process tool calls from model response.

        Args:
            chat_message: Model response with tool calls
            action_step: Current action step

        Yields:
            Tool execution results as step outputs
        """
        if not chat_message.tool_calls:
            return

        # Group tool calls for parallel execution
        parallel_calls = []
        sequential_calls = []

        for tool_call in chat_message.tool_calls:
            # For simplicity, execute all tools in parallel
            # In practice, you might want to analyze dependencies
            parallel_calls.append(tool_call)

        # Execute parallel calls
        if parallel_calls:
            tasks = []
            for tool_call in parallel_calls:
                tasks.append(self._execute_single_tool_call(tool_call))

            # Wait for all tool calls to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect observations and separate errors
            observations = []
            errors = []
            for tool_call, result in zip(parallel_calls, results):
                if isinstance(result, Exception):
                    # Handle exceptions from gather
                    error_msg = f"❌ ERROR in {tool_call.name}: {str(result)}"
                    errors.append(error_msg)
                    observations.append(error_msg)
                elif isinstance(result, dict) and result.get("_error"):
                    # Handle structured errors from tool execution
                    error_type = result.get("error_type", "UnknownError")
                    message = result.get("message", "Unknown error")
                    error_msg = f"❌ {error_type} in {tool_call.name}: {message}"
                    errors.append(error_msg)
                    observations.append(error_msg)
                    # Store traceback separately for debugging if needed
                    if result.get("traceback"):
                        tool_call.error_traceback = result["traceback"]
                else:
                    # Successful result
                    obs = str(result)
                    observations.append(f"✅ [{tool_call.name}]: {obs}")
                tool_call.result = result

            # Combine observations (errors are already marked)
            combined_observations = "\n\n".join(observations)
            action_step.observations = combined_observations

            # Yield observation
            yield StepOutput(
                output=combined_observations,
                observations=combined_observations,
                is_final_answer=False
            )

    async def _execute_single_tool_call(self, tool_call: ToolCall) -> Any:
        """Execute a single tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            Tool execution result (or dict with error info if failed)
        """
        try:
            # Parse arguments if they're a string
            if isinstance(tool_call.arguments, str):
                try:
                    arguments = json.loads(tool_call.arguments)
                except json.JSONDecodeError:
                    arguments = {"input": tool_call.arguments}
            else:
                arguments = tool_call.arguments

            # Execute tool
            result = await self.execute_tool_call(tool_call.name, arguments)
            return result

        except Exception as e:
            # Return structured error information
            return {
                "_error": True,
                "error_type": "ToolExecutionError",
                "tool_name": tool_call.name,
                "message": str(e),
                "traceback": traceback.format_exc()
            }

    def _is_final_answer(self, content: str, tool_calls: Optional[List[Any]] = None) -> bool:
        """Check if content represents a final answer.

        Args:
            content: Model output content
            tool_calls: Optional list of tool calls to check

        Returns:
            True if this is a final answer
        """
        # Check if FinalAnswerTool was called
        if tool_calls:
            for tool_call in tool_calls:
                if hasattr(tool_call, 'name') and tool_call.name == 'final_answer':
                    return True

        # Check if content contains FinalAnswerTool output
        if content and "FINAL_ANSWER_SUBMITTED:" in content:
            return True

        if not content:
            return False

        # Check for explicit final answer markers
        final_markers = [
            "final answer:",
            "final answer is",
            "the answer is:",
            "in conclusion:",
            "to summarize:",
            "</final_answer>",
            "[final answer]",
        ]

        content_lower = content.lower()
        for marker in final_markers:
            if marker in content_lower:
                return True

        # Check if there are no tool calls and content seems complete
        # This is a simple heuristic - could be improved
        if len(content) > 50 and not content.endswith("..."):
            # Check if it ends with proper punctuation
            if content.rstrip().endswith((".", "!", "?")):
                return True

        return False

    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool or managed agent call.

        Args:
            tool_name: Name of tool or managed agent
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        available_tools = {**self.tools, **self.managed_agents}

        if tool_name not in available_tools:
            # Return structured error for tool not found
            return {
                "_error": True,
                "error_type": "ToolNotFoundError",
                "tool_name": tool_name,
                "message": f"Tool '{tool_name}' not found",
                "available_tools": list(available_tools.keys())
            }

        tool = available_tools[tool_name]
        is_managed_agent = tool_name in self.managed_agents

        try:
            # Execute tool or managed agent
            if is_managed_agent:
                # Managed agents expect 'task' as main argument
                if "task" not in arguments:
                    # Try to extract task from various possible fields
                    task = arguments.get("input", arguments.get("query", str(arguments)))
                    arguments = {"task": task}
                result = await tool(**arguments)
            else:
                # Regular tools - handle both async and sync
                if asyncio.iscoroutinefunction(tool):
                    result = await tool(**arguments)
                else:
                    result = tool(**arguments)

            return result

        except Exception as e:
            # Return structured error information
            return {
                "_error": True,
                "error_type": "ToolExecutionError",
                "tool_name": tool_name,
                "message": str(e),
                "traceback": traceback.format_exc()
            }