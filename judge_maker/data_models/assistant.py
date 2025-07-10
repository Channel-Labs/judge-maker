import json
import yaml
from dataclasses import dataclass
from typing import Dict


@dataclass
class Assistant:
    name: str
    description: str

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Create an Assistant instance from a dictionary.
        
        Args:
            data: Dictionary containing assistant data
            
        Returns:
            Assistant instance
        """
        return cls(
            name=data['name'],
            description=data['description']
        )

    @property
    def prompt_object(self) -> dict:
        return {"name": self.name, "description": self.description}

    @property
    def prompt_format(self) -> str:
        return json.dumps(self.prompt_object, indent=4)


def load_assistant(yaml_file_path) -> Assistant:
    """Load assistant definition from YAML file and return Assistant object."""
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    return Assistant(
        name=data['name'],
        description=data['description']
    )