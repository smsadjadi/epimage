import yaml
import inspect
from pathlib import Path


class ConfigNamespace:
    def __init__(self, mapping: dict):
        for key, value in mapping.items():
            # Recursively wrap nested dicts
            if isinstance(value, dict):
                value = ConfigNamespace(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"


def load_config(config_path: Path = None, dotmap=False) -> ConfigNamespace:
    # If a path is provided, use it
    if config_path:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Provided config path does not exist: {config_path}")
    else:
        # Determine caller's directory
        caller_frame = inspect.stack()[1]
        start_dir = Path(caller_frame.filename).resolve().parent

        # 1. Look for YAML files in the current directory
        direct = sorted(start_dir.glob("*.yml"))
        if direct:
            config_path = direct[0]
        else:
            # 2. Recursively search subdirectories
            recursive = sorted(start_dir.rglob("*.yml"))
            if recursive:
                config_path = recursive[0]
            else:
                # 3. Walk up parent directories (non-recursive)
                for directory in start_dir.parents:
                    found = sorted(directory.glob("*.yml"))
                    if found:
                        config_path = found[0]
                        break
                else:
                    raise FileNotFoundError(
                        f"No .yml file found in {start_dir}, its subfolders, or any parent directories."
                    )

    # Print the config path found or provided
    print(f"Config file loaded from: {config_path}")

    # Load and wrap
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if dotmap:
        return ConfigNamespace(data)
    else:
        return data
