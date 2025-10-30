"""Tools module - Import all tools to trigger registration.

This module imports all tool modules to ensure their @register_tool() decorators
are executed, which registers them in the global TOOL registry.
"""

# Import all tool modules to trigger @register_tool() decorators
from . import default_tools
from . import planning_tool
from . import final_answer

# Import the TOOL registry for access
from ..core.registry import TOOL

# Optional: explicitly import tool functions for convenience
from .default_tools import (
    python_interpreter_tool,
    web_search_tool,
    file_reader_tool,
    web_scraper_tool,
    deep_researcher_tool,
    deep_analyzer_tool,
    browser_tool,
    screenshot_tool,
)

from .planning_tool import (
    planning_tool,
    analyze_and_plan,
)

from .final_answer import (
    final_answer_tool,
    validate_answer_tool,
)

# Export all tools
__all__ = [
    'TOOL',
    'python_interpreter_tool',
    'web_search_tool',
    'file_reader_tool',
    'web_scraper_tool',
    'deep_researcher_tool',
    'deep_analyzer_tool',
    'browser_tool',
    'screenshot_tool',
    'planning_tool',
    'analyze_and_plan',
    'final_answer_tool',
    'validate_answer_tool',
]