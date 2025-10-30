"""Agent progress monitoring utilities."""

import asyncio
import time
from typing import Any


async def monitor_agent_progress(agent: Any, logger: Any, task_id: str, stop_event: asyncio.Event) -> None:
    """Monitor and log agent steps in real-time without streaming.

    Args:
        agent: The agent instance to monitor
        logger: Logger instance for output
        task_id: Current task ID for context
        stop_event: Event to signal when to stop monitoring
    """
    seen_steps = 0
    last_activity_time = time.time()
    shown_waiting = False
    wait_threshold = 2.0  # Show "waiting" after 2 seconds of no activity

    while not stop_event.is_set():
        # Check if agent has memory and new steps
        if agent.memory and len(agent.memory.steps) > seen_steps:
            # Reset waiting indicator
            shown_waiting = False
            last_activity_time = time.time()

            # Log new steps
            for i, step in enumerate(agent.memory.steps[seen_steps:], seen_steps):

                # Handle different step types
                if hasattr(step, 'plan'):
                    # PlanningStep
                    plan_lines = step.plan.split('\n')
                    # Show first 3 lines of plan
                    plan_preview = '\n      '.join(plan_lines[:3])
                    if len(plan_lines) > 3:
                        plan_preview += f"\n      ... ({len(plan_lines) - 3} more lines)"
                    logger.info(f"  📋 Planning (Step {step.step_number}):", prefix="")
                    logger.info(f"      {plan_preview}", prefix="")

                elif hasattr(step, 'tool_calls'):  # ActionStep
                    step_num = step.step_number if hasattr(step, 'step_number') else i + 1

                    # Show timing if available
                    duration_str = ""
                    elapsed_str = ""
                    if hasattr(step, 'timing'):
                        if isinstance(step.timing, dict):
                            start = step.timing.get('start', 0)
                            end = step.timing.get('end')
                            if end:
                                duration = end - start
                                duration_str = f" [{duration:.1f}s]"
                            elif start:
                                # Step in progress - show elapsed time
                                elapsed = time.time() - start
                                elapsed_str = f" [{elapsed:.1f}s elapsed]"
                        elif hasattr(step.timing, 'duration') and step.timing.duration:
                            duration_str = f" [{step.timing.duration:.1f}s]"

                    # Determine step status
                    status_icon = "⏳"  # In progress
                    if hasattr(step, 'timing') and isinstance(step.timing, dict) and step.timing.get('end'):
                        status_icon = "✅"  # Completed
                    elif hasattr(step, 'error') and step.error:
                        status_icon = "❌"  # Error

                    # Check for errors first
                    if hasattr(step, 'error') and step.error:
                        logger.error(f"  ❌ Step {step_num} - Error{duration_str}:", prefix="")
                        logger.error(f"      Type: {step.error.error_type}", prefix="")
                        logger.error(f"      Message: {step.error.message}", prefix="")
                        if step.error.traceback:
                            # Show first few lines of traceback
                            tb_lines = step.error.traceback.split('\n')[:5]
                            logger.error(f"      Traceback:", prefix="")
                            for line in tb_lines:
                                if line.strip():
                                    logger.error(f"        {line}", prefix="")

                    # Show step header with status
                    elif not (hasattr(step, 'timing') and isinstance(step.timing, dict) and step.timing.get('end')):
                        # Step in progress
                        logger.info(f"  {status_icon} Step {step_num} - Processing{elapsed_str}:", prefix="")

                    # Show model output if available
                    elif hasattr(step, 'model_output') and step.model_output:
                        # Show we're thinking
                        logger.info(f"  🤔 Step {step_num} - Model reasoning{duration_str}:", prefix="")
                        # Show first 200 chars of model output
                        model_text = step.model_output.strip()
                        if len(model_text) > 200:
                            model_text = model_text[:200] + "..."
                        # Split into lines for better formatting
                        for line in model_text.split('\n')[:3]:
                            if line.strip():
                                logger.info(f"      {line}", prefix="")

                    # Show tool calls if present
                    if step.tool_calls:
                        tools_summary = []
                        for tc in step.tool_calls:
                            # Show tool name and key arguments
                            args_preview = ""
                            if hasattr(tc, 'arguments') and tc.arguments:
                                # Show first few key-value pairs
                                if isinstance(tc.arguments, dict):
                                    args_items = list(tc.arguments.items())[:2]
                                    args_strs = [f"{k}={repr(v)[:30]}" for k, v in args_items]
                                    if len(tc.arguments) > 2:
                                        args_strs.append(f"...+{len(tc.arguments)-2}")
                                    args_preview = f"({', '.join(args_strs)})"
                                else:
                                    args_preview = f"({repr(tc.arguments)[:50]})"
                            tool_name = tc.name if hasattr(tc, 'name') else str(tc)
                            tools_summary.append(f"{tool_name}{args_preview}")

                        logger.info(f"  🔧 Step {step_num} - Executing tools{duration_str if duration_str else elapsed_str}:", prefix="")
                        for tool_desc in tools_summary:
                            logger.info(f"      → {tool_desc}", prefix="")

                        # Show observations (tool results) with better formatting
                        if hasattr(step, 'observations') and step.observations:
                            # Handle different observation formats
                            if isinstance(step.observations, list):
                                # New format: list of dicts with tool_name, tool_args, result
                                for obs in step.observations:
                                    if isinstance(obs, dict):
                                        tool_name = obs.get('tool_name', 'Unknown')
                                        result = obs.get('result', 'Executing...')

                                        if result == "Executing...":
                                            # Tool is still running
                                            logger.info(f"      ⏳ {tool_name}: Executing...", prefix="")
                                        elif "Error" in str(result) or "❌" in str(result):
                                            # Tool had an error
                                            logger.error(f"      ❌ {tool_name}: Error", prefix="")
                                            result_lines = str(result).split('\n')[:3]
                                            for line in result_lines:
                                                if line.strip():
                                                    logger.error(f"         {line[:150]}", prefix="")
                                        else:
                                            # Tool succeeded
                                            logger.success(f"      ✅ {tool_name}: Complete", prefix="")
                                            result_lines = str(result).split('\n')[:3]
                                            for line in result_lines:
                                                if line.strip():
                                                    logger.info(f"         {line[:150]}", prefix="")
                            else:
                                # Old format: plain string
                                obs_text = str(step.observations).strip()
                                # Check if it contains errors
                                if "❌" in obs_text or obs_text.startswith("Error:"):
                                    # This observation contains errors
                                    obs_lines = obs_text.split('\n')
                                    logger.error(f"      ⚠️  Tool execution had errors:", prefix="")
                                    for line in obs_lines[:10]:  # Show more lines for errors
                                        if line.strip():
                                            if len(line) > 200:
                                                line = line[:200] + "..."
                                            logger.error(f"         {line}", prefix="")
                                    if len(obs_lines) > 10:
                                        logger.error(f"         ... ({len(obs_lines)-10} more lines)", prefix="")
                                else:
                                    # Show successful results
                                    obs_lines = obs_text.split('\n')
                                    logger.info(f"      📊 Result:", prefix="")
                                    for line in obs_lines[:5]:  # Show first 5 lines
                                        if line.strip():
                                            if len(line) > 150:
                                                line = line[:150] + "..."
                                            logger.info(f"         {line}", prefix="")
                                    if len(obs_lines) > 5:
                                        logger.info(f"         ... ({len(obs_lines)-5} more lines)", prefix="")

                    # Show token usage if available
                    if hasattr(step, 'token_usage') and step.token_usage:
                        tokens = step.token_usage
                        logger.info(f"      💰 Tokens: in={tokens.input_tokens}, out={tokens.output_tokens}, total={tokens.total_tokens}", prefix="")

                    # Check if this is a final answer
                    if hasattr(step, 'is_final_answer') and step.is_final_answer:
                        logger.success(f"  ✅ Final Answer Ready!", prefix="")

            seen_steps = len(agent.memory.steps)
        else:
            # No new steps - show waiting indicator if it's been a while
            elapsed = time.time() - last_activity_time
            if elapsed > wait_threshold and not shown_waiting:
                logger.info("  ⏳ Agent is processing...", prefix="", min_level=1)
                shown_waiting = True

        # Poll every 0.5 seconds
        await asyncio.sleep(0.5)