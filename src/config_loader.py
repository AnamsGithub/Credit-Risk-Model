import yaml
import os

class ConfigLoader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, key_path: str, default=None):
        """
        Get value from nested config using dot notation (e.g. 'scaling.base_score')
        """
        keys = key_path.split(".")
        val = self.config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val
