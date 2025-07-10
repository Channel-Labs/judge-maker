from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional
import json
from uuid import uuid4


class ROLE(Enum):
    user = auto()
    assistant = auto()

@dataclass
class Message:
    role: ROLE
    content: str
    timestamp: datetime
    message_id: str

    @property
    def prompt_format(self):
        return {"message_id": self.message_id, "role": self.role.name.lower(), "content": self.content}


@dataclass
class Conversation:
    id: str
    user_id: str
    messages: List[Message]

    def __hash__(self):
        return hash((self.id, self.user_id))

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.id == other.id and self.user_id == other.user_id

    @property
    def start_time(self) -> Optional[datetime]:
        if not self.messages:
            return None

        return min(message.timestamp for message in self.messages)

    @property
    def end_time(self) -> Optional[datetime]:
        if not self.messages:
            return None

        return max(message.timestamp for message in self.messages)

    @property
    def prompt_format(self):
        return [m.prompt_format for m in self.messages]


def load_conversations(jsonl_file_path) -> List[Conversation]:
    """Load conversations from JSONL file and return Conversation objects."""
    conversations = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                
                # Create Message objects
                messages = []
                for i, msg_data in enumerate(data['messages']):
                    role = ROLE.user if msg_data['role'] == 'user' else ROLE.assistant
                    message = Message(
                        role=role,
                        content=msg_data['content'],
                        timestamp=datetime.now(),
                        message_id=int(i)
                    )
                    messages.append(message)
                
                # Create Conversation object
                conversation = Conversation(
                    id=str(uuid4()),
                    user_id=f"user_{line_num}",
                    messages=messages
                )
                conversations.append(conversation)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return conversations

