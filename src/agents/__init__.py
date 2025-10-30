"""Agent module - Import all agents to trigger registration."""

# Import all agent classes to trigger @register_agent() decorators
from .general import GeneralAgent
from .planning import PlanningAgent
from .analyzer import DeepAnalyzerAgent
from .researcher import DeepResearcherAgent
from .browser import BrowserUseAgent

# Import agent creation functions
from .agent import create_agent, build_agent, prepare_response

__all__ = [
    'GeneralAgent',
    'PlanningAgent',
    'DeepAnalyzerAgent',
    'DeepResearcherAgent',
    'BrowserUseAgent',
    'create_agent',
    'build_agent',
    'prepare_response',
]
