"""Agent building and creation functions - wrapper around AgentFactory."""

from typing import Any, Dict, List, Optional
from .factory import AgentFactory
from .base import AsyncMultiStepAgent


async def build_agent(
    config: Any,
    agent_config: Dict[str, Any],
    model: Any,
    default_tools: Optional[Dict] = None,
    default_managed_agents: Optional[List] = None,
    **kwargs
) -> AsyncMultiStepAgent:
    """Build an individual agent from configuration.

    This is a compatibility wrapper around AgentFactory.

    Args:
        config: Main configuration object
        agent_config: Agent-specific configuration
        model: Model instance to use
        default_tools: Default tools registry (ignored, uses factory defaults)
        default_managed_agents: Default managed agents list
        **kwargs: Additional arguments

    Returns:
        Built agent instance
    """
    # Get agent type and map old naming to new
    agent_type = agent_config.get("type", "general_agent")
    # Remove _agent suffix if present
    if agent_type.endswith("_agent"):
        agent_type = agent_type[:-6]

    # Extract configuration parameters
    overrides = {
        'name': agent_config.get("name", agent_type),
        'max_steps': agent_config.get("max_steps", 10),
        'provide_run_summary': agent_config.get("provide_run_summary", False),
    }

    if agent_config.get("planning_interval"):
        overrides['planning_interval'] = agent_config["planning_interval"]

    if agent_config.get("tools"):
        overrides['tools'] = agent_config["tools"]

    if default_managed_agents:
        overrides['managed_agents'] = default_managed_agents

    if agent_config.get("template_path"):
        overrides['template_path'] = agent_config["template_path"]

    # Use factory to create agent
    return await AgentFactory.create(agent_type, model=model, **overrides, **kwargs)


async def create_agent(
    config: Any,
    default_tools: Optional[Dict] = None,
    model: Optional[Any] = None,
    **kwargs
) -> AsyncMultiStepAgent:
    """Create agent from configuration, supporting hierarchical agents.

    This is a compatibility wrapper around AgentFactory.

    Args:
        config: Configuration object with agent settings
        default_tools: Default tools registry (ignored, uses factory defaults)
        model: Model instance to use (builds from config if None)
        **kwargs: Additional arguments

    Returns:
        Created agent (either single or hierarchical)
    """
    # If model not provided, build from config
    if model is None:
        from ..core.models import MODEL, ModelConfig
        model_config = ModelConfig(**config.model)
        model = MODEL.build({"type": "claude_agent"}, config=model_config)

    # Check if using hierarchical agents
    use_hierarchical = getattr(config, "use_hierarchical_agent", False)

    if use_hierarchical:
        # Build managed (child) agents first
        managed_agents = []

        # Get main agent config
        main_agent_config = getattr(config, "agent_config", config.planning_agent_config)

        # Get list of managed agents
        managed_agent_names = main_agent_config.get("managed_agents", [])

        for agent_name in managed_agent_names:
            # Get config for this managed agent
            managed_config_attr = f"{agent_name}_config"

            if hasattr(config, managed_config_attr):
                managed_agent_config = getattr(config, managed_config_attr)
            else:
                print(f"Warning: Config for managed agent '{agent_name}' not found")
                continue

            # Build managed agent
            managed_agent = await build_agent(
                config,
                managed_agent_config,
                model,
                default_tools=default_tools,
                default_managed_agents=None,
                **kwargs
            )

            managed_agents.append(managed_agent)

        # Build main agent with managed agents
        agent = await build_agent(
            config,
            main_agent_config,
            model,
            default_tools=default_tools,
            default_managed_agents=managed_agents,
            **kwargs
        )

    else:
        # Single agent mode
        agent_config = getattr(config, "agent_config", {"type": "general_agent"})

        agent = await build_agent(
            config,
            agent_config,
            model,
            default_tools=default_tools,
            default_managed_agents=None,
            **kwargs
        )

    return agent


async def prepare_response(
    question: str,
    agent_memory: List[Any],
    reformulation_model: Optional[Any] = None
) -> str:
    """Prepare final response from agent output with optional reformulation.

    Args:
        question: Original question
        agent_memory: Agent execution memory/messages
        reformulation_model: Optional model for reformulation

    Returns:
        Cleaned final response
    """
    if not agent_memory:
        return "Unable to determine the answer."

    # Extract final answer from memory
    final_answer = None

    # Look for assistant messages with final answer
    for message in reversed(agent_memory):
        if hasattr(message, 'role') and message.role == "assistant":
            content = message.content
            content_lower = content.lower()

            # Use AnswerProcessor for better extraction
            from ..gaia.answer_processor import AnswerProcessor
            processed = AnswerProcessor.process_agent_response(content)

            # Check if extraction was successful with high confidence
            if processed["answer"] and processed["metadata"].get("confidence", 0) >= 0.7:
                final_answer = processed["answer"]
                break

            # Fallback to original logic for lower confidence
            elif "the answer is" in content_lower or "the answer to your question is" in content_lower:
                # Extract the answer after the marker
                import re
                # Try to extract the answer after "the answer is" or similar phrases
                patterns = [
                    r"the answer (?:to your question )?is[:\s]+([^\n.!]+)",
                    r"answer[:\s]+([^\n.!]+)",
                    r"\*\*([^*]+)\*\*"  # Extract bold text as potential answer
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        extracted = match.group(1).strip()
                        # Clean up the extracted answer
                        extracted = extracted.strip('*').strip('"').strip("'").strip()
                        if extracted:
                            final_answer = extracted
                            break
                if final_answer:
                    break
            elif any(marker in content_lower for marker in [
                "final answer:", "final_answer_submitted:", "in conclusion:",
                "task completed", "task complete",
                "already completed", "completed the task"
            ]):
                # Use AnswerProcessor for these markers too
                final_answer = processed["answer"] if processed["answer"] else content
                break

    if not final_answer:
        # Use last assistant message
        for message in reversed(agent_memory):
            if hasattr(message, 'role') and message.role == "assistant":
                final_answer = message.content
                break

    if not final_answer:
        return "Unable to determine the answer."

    # Optional reformulation with another model
    if reformulation_model:
        try:
            # Create reformulation prompt
            reformulation_prompt = f"""Based on the following agent output, extract and provide a clean, direct answer to the user's question.

User's Question: {question}

Agent Output:
{final_answer}

Instructions:
- Extract only the relevant final answer
- Remove any internal reasoning or tool calls
- Present the answer clearly and directly
- If the answer cannot be determined, say "Unable to determine"

Clean Answer:"""

            # Call reformulation model
            from ..core.memory import ChatMessage
            response = await reformulation_model(
                [ChatMessage(role="user", content=reformulation_prompt)],
                tools_to_call_from=None
            )

            if response and response.content:
                final_answer = response.content

        except Exception as e:
            print(f"Warning: Reformulation failed: {e}")
            # Keep original answer

    return final_answer