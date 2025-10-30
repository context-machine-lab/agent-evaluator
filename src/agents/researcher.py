"""Deep researcher agent specialized for iterative web research based on DeepResearchAgent patterns."""

from typing import Any, Dict, List, Optional

from .general import GeneralAgent
from ..core.registry import register_agent
from ..core.logger import logger


@register_agent()
class DeepResearcherAgent(GeneralAgent):
    """Agent specialized for conducting iterative, in-depth web searches and research."""

    def __init__(
        self,
        name: str = "deep_researcher_agent",
        model: Any = None,
        tools: Optional[Dict[str, Any]] = None,
        managed_agents: Optional[List] = None,
        max_steps: int = 5,  # Increased for iterative research
        planning_interval: Optional[int] = None,
        provide_run_summary: bool = True,
        prompt_templates: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize deep researcher agent with iterative research capabilities.

        Args:
            name: Agent name
            model: Language model instance
            tools: Dictionary of available tools (should include web search tools)
            managed_agents: Not used for this agent
            max_steps: Maximum execution steps (default 5 for iterative research)
            planning_interval: Steps between planning (None = no planning)
            provide_run_summary: Include execution summary
            prompt_templates: Custom prompt templates
        """
        # Set default researcher templates if not provided
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

        self.description = "A deep researcher agent that conducts iterative, multi-level web searches with insight extraction."

    def _get_default_templates(self) -> Dict:
        """Get default deep researcher agent templates."""
        return {
            "system_prompt": """You are a Deep Research Agent specialized in conducting iterative, multi-level web searches and research.

Your capabilities include:
1. Performing iterative searches with depth levels
2. Extracting insights and generating follow-up queries
3. Analyzing and synthesizing information from multiple sources
4. Fact-checking and verifying information
5. Providing comprehensive research reports with citations

Available tools:
- web_search: Search the web for information
- deep_researcher_tool: Conduct in-depth research with insight extraction
- python_interpreter: Perform calculations or data analysis

Iterative Research Methodology:
1. INITIAL SEARCH: Start with broad searches to understand the topic
2. INSIGHT EXTRACTION: Identify key insights and generate follow-up queries
3. DEPTH EXPLORATION: Pursue promising leads with targeted searches
4. CROSS-REFERENCE: Verify information from multiple sources
5. SYNTHESIS: Compile findings into a comprehensive report

For each search iteration:
- Extract 3-5 key insights
- Generate 2-3 follow-up queries for promising leads
- Score relevance (0-1) for prioritization
- Track exploration depth

Always:
- Cite your sources with URLs when possible
- Distinguish between facts and opinions
- Note any conflicting information found
- Provide confidence levels for findings
- Include context and background information""",

            "user_prompt": "",

            "task_instruction": """Research the following topic thoroughly:
{{task}}

Provide a comprehensive research report with:
1. Key findings and facts
2. Multiple perspectives (if applicable)
3. Sources and references
4. Any limitations or uncertainties""",

            "planning": None,  # No planning for focused research

            "managed_agent": {
                "task": """You're the deep researcher agent.

Research task from your manager:
---
{{task}}
---

Provide a thorough research report with:
### 1. Task outcome (short version):
[Brief summary of research findings]

### 2. Task outcome (extremely detailed version):
[Comprehensive research results with all data, sources, and analysis]

### 3. Additional context (if relevant):
[Methodology, limitations, further research recommendations]""",

                "report": """Research Report from {{name}}:

{{final_answer}}"""
            },

            "final_answer": {
                "prompt": "Compile your research findings into a comprehensive final report.",
                "format": "Structure your answer with clear sections, citations, and a conclusion."
            }
        }

    def _is_final_answer(self, content: str) -> bool:
        """Check if content represents a final research report.

        Args:
            content: Model output content

        Returns:
            True if this is a final answer
        """
        if not content:
            return False

        # Research-specific completion markers
        research_markers = [
            "research complete",
            "research findings:",
            "research report:",
            "## findings",
            "## conclusion",
            "## summary",
            "based on my research",
            "research indicates",
            "the research shows"
        ]

        content_lower = content.lower()
        for marker in research_markers:
            if marker in content_lower:
                # Check if content is substantial (not just a header)
                if len(content) > 200:
                    return True

        # Check for report-like structure
        has_sections = any(marker in content_lower for marker in ["##", "###", "**findings", "**conclusion"])
        has_substance = len(content) > 500
        has_citations = any(marker in content_lower for marker in ["source:", "reference:", "according to", "cited"])

        if has_sections and has_substance:
            return True

        if has_citations and has_substance:
            return True

        return super()._is_final_answer(content)

