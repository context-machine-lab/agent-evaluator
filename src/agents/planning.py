"""Planning agent for hierarchical agent coordination based on DeepResearchAgent patterns."""

from typing import Any, Dict, List, Optional, AsyncIterator
import time

from .general import GeneralAgent
from .base import AsyncMultiStepAgent, StepOutput
from ..core.memory import ActionStep, PlanningStep, ChatMessage, ToolCall
from ..core.registry import register_agent
from ..core.logger import logger


@register_agent()
class PlanningAgent(GeneralAgent):
    """Planning agent that coordinates child agents hierarchically with structured plan management."""

    def __init__(
        self,
        name: str = "planning_agent",
        model: Any = None,
        tools: Optional[Dict[str, Any]] = None,
        managed_agents: Optional[List[AsyncMultiStepAgent]] = None,
        max_steps: int = 20,
        planning_interval: Optional[int] = 5,
        provide_run_summary: bool = True,
        prompt_templates: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize planning agent.

        Args:
            name: Agent name
            model: Language model instance
            tools: Dictionary of available tools
            managed_agents: List of child agents to coordinate
            max_steps: Maximum execution steps
            planning_interval: Steps between re-planning
            provide_run_summary: Include execution summary
            prompt_templates: Custom prompt templates
        """
        # Set default planning agent templates if not provided
        if prompt_templates is None:
            prompt_templates = self._get_default_templates()

        # Ensure planning_tool is available
        if tools is None:
            tools = {}
        if "planning_tool" not in tools:
            # Will be registered via registry
            pass

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

        self.description = "A planning agent that coordinates other agents to complete complex tasks."

    def _get_default_templates(self) -> Dict:
        """Get default planning agent templates."""
        return {
            "system_prompt": """You are a Planning Agent that coordinates tasks to solve complex problems.

AVAILABLE TOOLS (All Claude Code SDK Tools):

**Task Management:**
- **Task** - Launch specialized sub-agents for complex multi-step tasks
- **TodoWrite** - Create and manage todo lists to track progress
- **ExitPlanMode** - Exit planning mode when ready to execute

**Answer Submission:**
- **final_answer** - Submit your final answer (cleaned and validated automatically)
- **validate_answer** - Validate answer format before submission

**File Operations:**
- **Read** - Read files from the filesystem
- **Write** - Write/create new files
- **Edit** - Edit existing files with string replacement
- **Glob** - Find files by glob patterns (e.g., "**/*.py")
- **Grep** - Search file contents with regex patterns
- **NotebookEdit** - Edit Jupyter notebooks

**Code Execution:**
- **Bash** - Execute bash commands (can run Python with `uv run python -c "code"`)
- **BashOutput** - Get output from background bash processes
- **KillBash** - Kill background bash shells

**Web & Research:**
- **WebSearch** - Search the web for information
- **WebFetch** - Fetch and analyze web pages with AI assistance

**MCP Resources:**
- **ListMcpResources** - List available MCP resources
- **ReadMcpResource** - Read MCP resources

Your workflow:
1. Use TodoWrite to create a structured plan with tasks
2. Use WebSearch/WebFetch for research
3. Use Glob/Grep to search files and content
4. Use Read to examine files, Edit/Write to modify them
5. Use Bash to run code (use run_in_background=true for long tasks, then BashOutput to check)
6. Use Task to delegate complex sub-tasks to specialized agents
7. Synthesize all results into a comprehensive final answer

CRITICAL: When you have completed the task and are ready to provide your final conclusion:
- Use the **final_answer** tool to submit your answer
- The tool will clean and validate your answer automatically
- Provide ONLY the answer value without any prefixes or explanations
- Example: final_answer(answer="42") or final_answer(answer="Paris")
- If unsure about format, use validate_answer tool first

IMPORTANT: Only use the tools listed above.""",

            "user_prompt": "",

            "task_instruction": """Complete the following task by coordinating your team:
{{task}}

Important guidelines:
- If the task involves analyzing an ATTACHED FILE, a URL, performing CALCULATIONS, or playing GAME, use deep_analyzer_agent
- If the task involves interacting with web pages or conducting web searches, start with browser_use_agent and follow up with deep_researcher_agent if needed
- Always provide detailed task descriptions when delegating
- Synthesize team outputs into a comprehensive final answer
- ALWAYS use the final_answer tool to submit your answer when complete
- The final_answer tool will clean and validate your answer automatically""",

            "planning": {
                "prompt": """Based on the task and progress so far, create a plan for the next steps.

Consider:
1. What has been accomplished?
2. What still needs to be done?
3. Which team member should handle the next step?
4. Are there any dependencies between tasks?

Provide a clear, numbered plan."""
            },

            "managed_agent": {
                "task": """You're a helpful agent named '{{name}}'.

You have been submitted this task by your manager:
---
{{task}}
---

Your final answer MUST contain these parts:
### 1. Task outcome (short version):
[Brief summary of what was accomplished]

### 2. Task outcome (extremely detailed version):
[Detailed explanation with all findings, data, and analysis]

### 3. Additional context (if relevant):
[Any additional information, caveats, or recommendations]""",

                "report": """Report from {{name}}:

{{final_answer}}"""
            },

            "final_answer": {
                "prompt": "Based on all the work done, provide a comprehensive final answer to the original task.",
                "format": "Provide a clear, well-structured answer that directly addresses the user's request."
            }
        }

    async def _generate_planning_step(
        self,
        task: str,
        images: Optional[List[Any]] = None
    ) -> Optional[PlanningStep]:
        """Generate a planning step to guide execution.

        Args:
            task: Current task
            images: Optional images

        Returns:
            Planning step with the plan
        """
        if not self.prompt_templates.get("planning"):
            return None

        try:
            # Prepare planning prompt
            planning_prompt = self.prompt_templates["planning"]["prompt"]

            # Get current memory context
            memory_context = await self.write_memory_to_messages(summary_mode=True)

            # Add planning instruction to context
            memory_context.append(ChatMessage(
                role="user",
                content=planning_prompt
            ))

            # Generate plan using model
            start_time = time.time()
            response = await self.model(
                memory_context,
                tools_to_call_from=None  # No tools during planning
            )
            end_time = time.time()

            # Create planning step
            planning_step = PlanningStep(
                step_number=self.step_number,
                plan=response.content,
                timing={"start": start_time, "end": end_time}
            )

            return planning_step

        except Exception as e:
            # Log error but don't fail the execution
            print(f"Planning generation failed: {e}")
            return None

    def _analyze_task_type(self, task: str) -> str:
        """Analyze task to determine which agent should handle it.

        Args:
            task: Task description

        Returns:
            Suggested agent name
        """
        task_lower = task.lower()

        # Check for file analysis tasks
        if any(keyword in task_lower for keyword in [
            "file", "attached", "analyze", "calculate", "computation",
            "game", "puzzle", "math", "equation", "formula"
        ]):
            return "deep_analyzer_agent"

        # Check for web interaction tasks
        if any(keyword in task_lower for keyword in [
            "website", "web page", "click", "browse", "navigate",
            "form", "button", "interact"
        ]):
            return "browser_use_agent"

        # Check for research tasks
        if any(keyword in task_lower for keyword in [
            "search", "research", "find", "look up", "information",
            "learn", "discover", "investigate"
        ]):
            return "deep_researcher_agent"

        # Default to researcher for general queries
        return "deep_researcher_agent"

    async def _step_stream(self, action_step: ActionStep) -> AsyncIterator[StepOutput]:
        """Execute a step with intelligent agent selection.

        Args:
            action_step: Step to execute

        Yields:
            Step outputs
        """
        # Use parent implementation but with task analysis
        async for output in super()._step_stream(action_step):
            yield output

    def _is_final_answer(self, content: str) -> bool:
        """Check if content represents a final answer.

        For planning agent, we look for explicit completion signals.

        Args:
            content: Model output content

        Returns:
            True if this is a final answer
        """
        if not content:
            return False

        # Planning agent specific markers
        final_markers = [
            "final answer:",
            "task completed",
            "task complete",
            "all steps completed",
            "objective achieved",
            "goal accomplished",
            "## final answer",
            "## conclusion",
            "## summary",
            "completed the task",
            "already completed",
            "the answer is",
            "the answer to your question is",
        ]

        content_lower = content.lower()
        for marker in final_markers:
            if marker in content_lower:
                return True

        # Check if all managed agents have been used and no more delegation
        # This is indicated by lack of tool call intent
        if "delegate" not in content_lower and "use" not in content_lower:
            # Check for summary-like content
            if any(word in content_lower for word in ["therefore", "in conclusion", "to summarize", "overall"]):
                return True

        # Fallback: Check if there are no tool calls and content seems complete
        # This helps catch cases where the agent has answered but didn't use explicit markers
        if len(content) > 50 and not content.endswith("..."):
            # Check if it ends with proper punctuation
            if content.rstrip().endswith((".", "!", "?")):
                # Additional check for planning agent: ensure it's not mid-planning
                if not any(phrase in content_lower for phrase in
                          ["next step", "will now", "let me", "i'll", "going to", "about to"]):
                    return True

        return False

    async def _handle_max_steps_reached(
        self,
        task: str,
        images: Optional[List[Any]] = None
    ) -> str:
        """Handle case when max steps is reached without final answer.

        For planning agent, try to extract the best answer from memory.

        Args:
            task: Original task
            images: Optional images

        Returns:
            Best available answer from memory
        """
        # Try to extract the best answer from memory
        if self.memory and self.memory.steps:
            # Look for the most recent model output that could be an answer
            best_answer = None

            # Traverse steps in reverse to find most recent substantial output
            for step in reversed(self.memory.steps):
                if hasattr(step, 'model_output') and step.model_output:
                    output = step.model_output.strip()

                    # Check if this looks like a potential answer
                    if len(output) > 50:
                        # Prioritize outputs with answer-like keywords
                        output_lower = output.lower()
                        answer_keywords = [
                            "the answer", "answer is", "result", "conclusion",
                            "found that", "determined", "shows that", "indicates",
                            "therefore", "thus", "hence", "egalitarian", "hierarchical"
                        ]

                        if any(keyword in output_lower for keyword in answer_keywords):
                            best_answer = output
                            break

                        # If no answer keywords but it's substantial, keep it as fallback
                        if not best_answer and len(output) > 100:
                            best_answer = output

            if best_answer:
                # Extract the core answer if we can identify it
                return self._extract_answer_from_content(best_answer)

        # Fallback to parent implementation if no good answer found
        return await super()._handle_max_steps_reached(task, images)

    def _extract_answer_from_content(self, content: str) -> str:
        """Extract the answer from model output content.

        Args:
            content: Model output content

        Returns:
            Extracted answer
        """
        # Look for explicit answer patterns
        patterns = [
            r"(?:final answer|the answer)(?:\s+is)?:?\s*(.+?)(?:\.|$)",
            r"(?:therefore|thus|hence),?\s+(.+?)(?:\.|$)",
            r"(?:found|determined|discovered)\s+that\s+(.+?)(?:\.|$)",
        ]

        import re
        content_lower = content.lower()

        for pattern in patterns:
            match = re.search(pattern, content_lower, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer
                answer = answer.split('\n')[0]  # Take first line only
                if answer:
                    return f"FINAL ANSWER: {answer}"

        # If no pattern matches but content looks complete, return it with prefix
        if content.rstrip().endswith((".", "!", "?")):
            # Take the last complete sentence as the answer
            sentences = content.split('.')
            if sentences:
                last_sentence = sentences[-2].strip() if len(sentences) > 1 else sentences[-1].strip()
                if last_sentence:
                    return f"FINAL ANSWER: {last_sentence}"

        # Return the full content as last resort
        return f"FINAL ANSWER: {content}"