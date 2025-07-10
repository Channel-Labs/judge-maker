from typing import Dict, List
import yaml

from dataclasses import dataclass


@dataclass
class CharacterCard:
    name: str
    description: str
    personality: str
    scenario: str
    summary: str

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Create a CharacterCard instance from a dictionary.
        
        Args:
            data: Dictionary containing character data
            
        Returns:
            CharacterCard instance
        """
        return cls(
            name=data['name'],
            description=data['description'],
            personality=data['personality'],
            scenario=data['scenario'],
            summary=data['summary']
        )


def load_user_personas(yaml_file_path) -> List[CharacterCard]:
    """Load user personas from YAML file and return list of CharacterCard objects."""
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    # Create CharacterCard objects for each user persona
    user_personas = []
    for user_data in data['users']:
        user_persona = CharacterCard.from_dict(user_data)
        user_personas.append(user_persona)
    
    return user_personas


