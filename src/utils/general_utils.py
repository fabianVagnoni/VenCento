import yaml

def _load_config(config_path):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading config at {config_path}: {e}")
        return {}