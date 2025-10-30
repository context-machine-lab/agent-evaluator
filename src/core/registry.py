"""Registry system for agents, tools, and models based on DeepResearchAgent patterns."""

from typing import Any, Dict, Optional, Type, Callable
import inspect


class Registry:
    """Generic registry for dynamic component registration and instantiation."""

    def __init__(self, name: str):
        """Initialize registry.

        Args:
            name: Registry name (e.g., "AGENT", "TOOL", "MODEL")
        """
        self.name = name
        self._modules: Dict[str, Any] = {}

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False
    ) -> Callable:
        """Decorator to register a module.

        Args:
            name: Module name (defaults to class name)
            force: Force registration even if name exists

        Returns:
            Decorator function
        """
        def decorator(cls_or_func):
            module_name = name or cls_or_func.__name__

            # Convert class name to snake_case if no name provided
            if name is None and inspect.isclass(cls_or_func):
                # Convert CamelCase to snake_case
                module_name = ''.join(
                    ['_' + c.lower() if c.isupper() else c for c in cls_or_func.__name__]
                ).lstrip('_')

            if module_name in self._modules and not force:
                raise ValueError(
                    f"Module '{module_name}' already registered in {self.name} registry"
                )

            self._modules[module_name] = cls_or_func
            return cls_or_func

        return decorator

    def get(self, name: str) -> Any:
        """Get a registered module.

        Args:
            name: Module name

        Returns:
            Registered module

        Raises:
            KeyError: If module not found
        """
        if name not in self._modules:
            raise KeyError(
                f"Module '{name}' not found in {self.name} registry. "
                f"Available modules: {list(self._modules.keys())}"
            )
        return self._modules[name]

    def build(
        self,
        cfg: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Build/instantiate a module from configuration.

        Args:
            cfg: Configuration dictionary with 'type' field
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Instantiated module
        """
        if isinstance(cfg, str):
            # If cfg is a string, use it as the type
            cfg = {"type": cfg}

        cfg = cfg.copy()  # Don't modify original config

        # Get module type
        module_type = cfg.pop("type", None)
        if module_type is None:
            raise ValueError("Configuration must include 'type' field")

        # Get the module class/function
        module = self.get(module_type)

        # Merge config with kwargs (kwargs override config)
        combined_kwargs = {**cfg, **kwargs}

        # Instantiate or call the module
        if inspect.isclass(module):
            # It's a class, instantiate it
            return module(*args, **combined_kwargs)
        else:
            # It's a function, call it
            return module(*args, **combined_kwargs)

    def list(self) -> list:
        """List all registered modules.

        Returns:
            List of registered module names
        """
        return list(self._modules.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a module is registered.

        Args:
            name: Module name

        Returns:
            True if registered
        """
        return name in self._modules

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}Registry({len(self._modules)} modules: {self.list()})"


# Create global registries
AGENT = Registry("AGENT")
TOOL = Registry("TOOL")
MODEL = Registry("MODEL")
DATASET = Registry("DATASET")


# Helper function for building agents with tools
def build_agent_with_tools(
    agent_config: Dict[str, Any],
    tool_configs: Optional[list] = None,
    **kwargs
) -> Any:
    """Build an agent with configured tools.

    Args:
        agent_config: Agent configuration
        tool_configs: List of tool configurations
        **kwargs: Additional arguments for agent

    Returns:
        Instantiated agent with tools
    """
    # Build tools first
    tools = {}
    if tool_configs:
        for tool_cfg in tool_configs:
            if isinstance(tool_cfg, str):
                tool_cfg = {"type": tool_cfg}
            tool_name = tool_cfg.get("name", tool_cfg["type"])
            tools[tool_name] = TOOL.build(tool_cfg)

    # Build agent with tools
    return AGENT.build(agent_config, tools=tools, **kwargs)


# Utility decorators for common registration patterns
def register_agent(name: Optional[str] = None, force: bool = False):
    """Decorator to register an agent class.

    Args:
        name: Agent name (defaults to snake_case of class name)
        force: Force registration

    Example:
        @register_agent()
        class PlanningAgent(AsyncMultiStepAgent):
            ...
    """
    return AGENT.register_module(name=name, force=force)


def register_tool(name: Optional[str] = None, force: bool = False):
    """Decorator to register a tool.

    Args:
        name: Tool name (defaults to function/class name)
        force: Force registration

    Example:
        @register_tool()
        async def python_interpreter_tool(**kwargs):
            ...
    """
    return TOOL.register_module(name=name, force=force)


def register_model(name: Optional[str] = None, force: bool = False):
    """Decorator to register a model.

    Args:
        name: Model name
        force: Force registration

    Example:
        @register_model("claude-sonnet-4-5-20250929")
        class ClaudeOpusModel:
            ...
    """
    return MODEL.register_module(name=name, force=force)


def register_dataset(name: Optional[str] = None, force: bool = False):
    """Decorator to register a dataset.

    Args:
        name: Dataset name
        force: Force registration

    Example:
        @register_dataset()
        class GAIADataset:
            ...
    """
    return DATASET.register_module(name=name, force=force)