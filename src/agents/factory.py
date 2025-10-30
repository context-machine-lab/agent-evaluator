"""Agent factory for simplified agent creation."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml


class AgentFactory:
    """Factory for creating agents with simplified configuration."""

    # Default configurations for each agent type
    # Using ALL Claude Agent SDK tool names
    # Available tools: Task, Bash, Edit, Read, Write, Glob, Grep, NotebookEdit, WebFetch,
    # WebSearch, TodoWrite, BashOutput, KillBash, ExitPlanMode, ListMcpResources, ReadMcpResource
    AGENT_DEFAULTS = {
        'general': {
            'max_steps': 20,
            'tools': [
                'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
                'WebSearch', 'WebFetch',
                'TodoWrite', 'BashOutput', 'KillBash',
                'NotebookEdit'
            ],
            'provide_run_summary': False,
        },
        'planning': {
            'max_steps': 50,
            'tools': [
                'Task',  # For launching sub-agents
                'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
                'WebSearch', 'WebFetch',
                'TodoWrite', 'BashOutput', 'KillBash',
                'NotebookEdit', 'ExitPlanMode',
                'ListMcpResources', 'ReadMcpResource'
            ],
            'provide_run_summary': True,
            'planning_interval': 10,
        },
        'researcher': {
            'max_steps': 10,
            'tools': [
                'Bash', 'Read', 'Write', 'Glob', 'Grep',
                'WebSearch', 'WebFetch',
                'TodoWrite', 'BashOutput', 'KillBash'
            ],
            'provide_run_summary': True,
        },
        'analyzer': {
            'max_steps': 10,
            'tools': [
                'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
                'TodoWrite', 'BashOutput', 'KillBash',
                'NotebookEdit'
            ],
            'provide_run_summary': True,
        },
        'browser': {
            'max_steps': 10,
            'tools': [
                'Bash', 'Read', 'Write',
                'WebFetch', 'WebSearch',
                'TodoWrite', 'BashOutput', 'KillBash'
            ],
            'provide_run_summary': True,
        },
    }

    @classmethod
    async def create(cls, agent_type: str, model: Optional[Any] = None, **overrides) -> Any:
        """Create a fresh agent instance.

        Args:
            agent_type: Type of agent to create
            model: Optional model instance (builds from config if None)
            **overrides: Configuration overrides

        Returns:
            Configured agent instance
        """
        # Get default configuration
        defaults = cls.AGENT_DEFAULTS.get(agent_type, {})
        config = {**defaults, **overrides}

        # Load tools
        tools = cls._load_tools(config.get('tools', []))

        # Get model
        if model is None:
            model = await cls._get_model(config)

        # Import the appropriate agent class
        agent_class = cls._get_agent_class(agent_type)

        # Create and return agent instance
        return agent_class(
            name=config.get('name', agent_type),
            model=model,
            tools=tools,
            max_steps=config.get('max_steps', 20),
            provide_run_summary=config.get('provide_run_summary', False),
            **{k: v for k, v in config.items() if k not in ['name', 'model', 'tools', 'max_steps', 'provide_run_summary']}
        )

    @staticmethod
    def _load_tools(tool_names: List[str]) -> Dict[str, Any]:
        """Load tools by name.

        For SDK tools (Bash, WebSearch, Read, Write, WebFetch, Task, etc.),
        we just return the tool names as-is. The SDK will handle them.

        Args:
            tool_names: List of SDK tool names

        Returns:
            Dict mapping tool names to themselves (SDK handles execution)
        """
        # For SDK tools, we just need the names - the SDK handles everything
        # Return a dict with tool names as both keys and values
        tools_dict = {name: name for name in tool_names}
        return tools_dict

    @staticmethod
    async def _get_model(config: Dict[str, Any]) -> Any:
        """Get model instance based on configuration.

        Args:
            config: Agent configuration

        Returns:
            Model instance
        """
        from ..core.models import MODEL, ModelConfig

        model_id = config.get('model_id', config.get('model', 'claude-sonnet-4-5-20250929'))
        model_config = ModelConfig(
            model_id=model_id,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 4096)
        )
        return MODEL.build({"type": "claude_agent"}, config=model_config)

    @staticmethod
    def _get_agent_class(agent_type: str) -> Any:
        """Get agent class by type.

        Args:
            agent_type: Type of agent

        Returns:
            Agent class
        """
        # Import agent classes
        if agent_type == 'general':
            from .general import GeneralAgent
            return GeneralAgent
        elif agent_type == 'planning':
            from .planning import PlanningAgent
            return PlanningAgent
        elif agent_type == 'researcher':
            from .researcher import DeepResearcherAgent
            return DeepResearcherAgent
        elif agent_type == 'analyzer':
            from .analyzer import DeepAnalyzerAgent
            return DeepAnalyzerAgent
        elif agent_type == 'browser':
            from .browser import BrowserUseAgent
            return BrowserUseAgent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    @classmethod
    def list_available_agents(cls) -> List[str]:
        """List all available agent types.

        Returns:
            List of agent type names
        """
        return list(cls.AGENT_DEFAULTS.keys())