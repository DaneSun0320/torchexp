"""

"""
import yaml

class ConfigParser:
    def __init__(self, path):
        self.path = path
        self.config = self._read_config()

    def _read_config(self):
        with open(self.path, "r") as f:
            return yaml.safe_load(f)

    def get_config(self):
        print(self.config)



