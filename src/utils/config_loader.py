"""Configuration loader utilities."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration object with dot notation access."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize config with dictionary data."""
        self._data = data

    def __getattr__(self, key: str) -> Any:
        """Access config values with dot notation."""
        value = self._data.get(key)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        """Access config values with bracket notation."""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        return self._data

    @property
    def agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self._data.get('agent', {})

    @property
    def model_configs(self) -> Dict[str, Any]:
        """Get model configurations."""
        return self._data.get('models', {})

    @property
    def save_path(self) -> Path:
        """Get results save path."""
        return Path(self._data.get('output', {}).get('save_path', 'outputs/results.jsonl'))

    @property
    def max_tasks(self) -> Optional[int]:
        """Get maximum number of tasks."""
        return self._data.get('dataset', {}).get('max_tasks')

    @property
    def filter_completed(self) -> bool:
        """Get whether to filter completed tasks."""
        return self._data.get('execution', {}).get('filter_completed', True)

    @property
    def concurrency(self) -> int:
        """Get concurrency level."""
        return self._data.get('execution', {}).get('concurrency', 4)

    @property
    def use_local_proxy(self) -> bool:
        """Get whether to use local proxy."""
        return self._data.get('execution', {}).get('use_local_proxy', False)

    @property
    def use_hierarchical_agent(self) -> bool:
        """Get whether to use hierarchical agent."""
        return self._data.get('agent', {}).get('use_hierarchical', False)

    @property
    def response_preparation(self) -> Config:
        """Get response preparation config."""
        return Config(self._data.get('response_preparation', {}))

    @property
    def dataset(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self._data.get('dataset', {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._data.get('evaluation', {})

    @property
    def pretty_text(self) -> str:
        """Get pretty printed configuration."""
        return yaml.dump(self._data, default_flow_style=False)


def load_config(config_path: str, args: Optional[Any] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        args: Optional command line arguments to override config

    Returns:
        Config object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML configuration
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Override with command line arguments if provided
    if args:
        if hasattr(args, 'max_tasks') and args.max_tasks is not None:
            data['dataset']['max_tasks'] = args.max_tasks
        if hasattr(args, 'split') and args.split:
            data['dataset']['split'] = args.split
        if hasattr(args, 'batch_size') and args.batch_size:
            data['execution']['concurrency'] = args.batch_size
        if hasattr(args, 'output') and args.output:
            data['output']['save_path'] = args.output

    return Config(data)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    result = {}
    for config in configs:
        _deep_merge(result, config)
    return result


def _deep_merge(dest: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Deep merge source dict into destination dict.

    Args:
        dest: Destination dictionary (modified in place)
        src: Source dictionary to merge
    """
    for key, value in src.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
            _deep_merge(dest[key], value)
        else:
            dest[key] = value
