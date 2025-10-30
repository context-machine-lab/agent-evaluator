"""Browser use agent specialized for web interaction based on DeepResearchAgent patterns."""

from typing import Any, Dict, List, Optional

from .general import GeneralAgent
from ..core.registry import register_agent


@register_agent()
class BrowserUseAgent(GeneralAgent):
    """Agent specialized for interacting with web pages and browser automation."""

    def __init__(
        self,
        name: str = "browser_use_agent",
        model: Any = None,
        tools: Optional[Dict[str, Any]] = None,
        managed_agents: Optional[List] = None,
        max_steps: int = 5,
        planning_interval: Optional[int] = None,
        provide_run_summary: bool = True,
        prompt_templates: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize browser use agent.

        Args:
            name: Agent name
            model: Language model instance
            tools: Dictionary of available tools (should include browser tools)
            managed_agents: Not used for this agent
            max_steps: Maximum execution steps (default 5 for web interactions)
            planning_interval: Steps between planning (None = no planning)
            provide_run_summary: Include execution summary
            prompt_templates: Custom prompt templates
        """
        # Set default browser templates if not provided
        if prompt_templates is None:
            prompt_templates = self._get_default_templates()

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

        self.description = "A browser automation agent that interacts with web pages and performs web-based tasks."

    def _get_default_templates(self) -> Dict:
        """Get default browser use agent templates."""
        return {
            "system_prompt": """You are a Browser Use Agent specialized in web page interaction and automation.

Your capabilities include:
1. Navigating to web pages
2. Clicking buttons and links
3. Filling out forms
4. Extracting information from web pages
5. Taking screenshots
6. Handling dynamic web content

Available tools:
- browser_tool: Control browser and interact with web pages
- web_scraper: Extract structured data from web pages
- screenshot: Capture screenshots of web pages

Interaction methodology:
1. Navigate to the target page
2. Wait for page to load completely
3. Identify elements to interact with
4. Perform required actions (click, type, select)
5. Verify actions were successful
6. Extract or capture results

Always:
- Handle errors gracefully (page not found, elements missing)
- Wait for dynamic content to load
- Verify successful completion of actions
- Capture evidence (screenshots) when relevant""",

            "user_prompt": "",

            "task_instruction": """Complete the following web interaction task:
{{task}}

Navigate to the specified pages, interact as needed, and extract/capture the required information.""",

            "planning": None,  # No planning for focused interactions

            "managed_agent": {
                "task": """You're the browser use agent.

Web interaction task from your manager:
---
{{task}}
---

Complete the task and report:
### 1. Task outcome (short version):
[Brief summary of what was accomplished]

### 2. Task outcome (extremely detailed version):
[Step-by-step account of all interactions, extracted data, and results]

### 3. Additional context (if relevant):
[Any issues encountered, alternative approaches, screenshots taken]""",

                "report": """Browser Interaction Report from {{name}}:

{{final_answer}}"""
            },

            "final_answer": {
                "prompt": "Summarize the web interaction results.",
                "format": "Provide a clear account of actions taken and information gathered."
            }
        }

    def _is_final_answer(self, content: str) -> bool:
        """Check if content represents completed web interaction.

        Args:
            content: Model output content

        Returns:
            True if this is a final answer
        """
        if not content:
            return False

        # Browser-specific completion markers
        browser_markers = [
            "task complete",
            "interaction complete",
            "successfully",
            "extracted",
            "captured",
            "screenshot taken",
            "form submitted",
            "page loaded",
            "data collected",
            "## results",
            "## extracted data"
        ]

        content_lower = content.lower()
        for marker in browser_markers:
            if marker in content_lower:
                # Check if it's not just status update
                if len(content) > 100 and not content_lower.endswith("..."):
                    return True

        # Check for extraction results
        has_data = any(marker in content for marker in ["{", "[", ":", "="])
        has_completion = any(word in content_lower for word in ["complete", "done", "finished", "extracted"])

        if has_data and has_completion:
            return True

        return super()._is_final_answer(content)