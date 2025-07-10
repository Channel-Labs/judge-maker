import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import io
import os
import random
import yaml
from uuid import uuid4
from tqdm import tqdm
import requests

from openai import OpenAI

from data_models.assistant import Assistant
from data_models.character_card import CharacterCard
from data_models.conversation import Conversation, Message, ROLE
from llm_queries.llm_query import OpenAIModelProvider
from llm_queries.ground_truth_judge_query import GroundTruthJudgeQuery
from llm_queries.create_grading_rubric_query import CreateGradingRubricQuery

openai_client = OpenAI()
openai_model_provider = OpenAIModelProvider(OpenAI())


def load_assistant_personas(yaml_file_path):
    """Load assistant personas from YAML file and return assistant and user personas."""
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    # Create Assistant object
    assistant_data = data['assistant']
    assistant = Assistant(
        name=assistant_data['name'],
        description=assistant_data['description']
    )
    
    # Create CharacterCard objects for each user persona
    user_personas = []
    for user_data in data['users']:
        user_persona = CharacterCard.from_dict(user_data)
        user_personas.append(user_persona)
    
    return assistant, user_personas


def load_conversations(jsonl_file_path):
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


def create_fine_tuning_job(training_file_id, validation_file_id):
    """
    Create a fine-tuning job using OpenAI's API.
    
    Returns:
        dict: Response from the API
    """
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # API endpoint
    url = "https://api.openai.com/v1/fine_tuning/jobs"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    judge_grader = {
      "type": "score_model",
      "name": "judge_grader",
      "input": [
            {
                "role": "user",
                "content": """Assess the assistant's performance in the following conversation on a scale of 0 to 100, based on the grading rubric provided.

### Grading Rubric
{{ sample.output_json.grading_rubric }}

### Assistant
{
    "name": "{{ item.assistant_name }}",
    "description": "{{ item.assistant_description }}"
}

### Conversation
{{ item.conversation_str }}        
"""
          }
      ],
      "model": "o3-2025-04-16",
      "range": [0, 100],
      "sampling_params": {
          "reasoning_effort": "medium"
      },
  }

    passthrough_grader = {
        "type": "python",
        "name": "passthrough_grader_v2",
        "source": "def grade(sample, item):\n    return int(item['expected_judge_score'])",
        "image_tag": "2025-05-08"
    }

    multi_grader = {
        "type": "multi",
        "graders": {
            "actual_score": judge_grader,
            "expected_score": passthrough_grader
        },
        "calculate_output": "(100 - abs(expected_score - actual_score)) / 100"
    }
        
    response_schema = CreateGradingRubricQuery(
        model_provider=None,
        model_id=None,
        assistant=None
    ).response_schema()
    
    # Request payload
    data = {
        "training_file": training_file_id,
        "validation_file": validation_file_id,
        "model": "o4-mini-2025-04-16",
        "seed": 42,
        "method": {
            "type": "reinforcement",
            "reinforcement": {
                "grader": multi_grader,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": response_schema
                    }
                },
                "hyperparameters": {
                    "reasoning_effort": "medium",
                    "batch_size": 6,
                    "eval_interval": 4,
                    "n_epochs": 2
                }
            }
        }
    }

    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating fine-tuning job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except:
                print(f"Response text: {e.response.text}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-personas-file", type=str, required=True)
    parser.add_argument("--conversations-file", type=str, required=True)
    parser.add_argument("--num-attempts-per-conversation", type=int, default=5)
    parser.add_argument("--ground-truth-judge-model-id", type=str, default="o3")
    parser.add_argument("--rubric-generator-model-id", type=str, default="o4-mini")
    parser.add_argument("--validation_fraction", type=float, default=0.25)
    args = parser.parse_args()
    
    # Load assistant personas and conversations
    assistant, user_personas = load_assistant_personas(args.assistant_personas_file)
    conversations = load_conversations(args.conversations_file)

    grading_rubric_template_prompt = CreateGradingRubricQuery(
        model_provider=None,
        model_id=None,
        assistant=assistant,
    ).generate_prompt()

    results = []
    
    print(f"Loaded {len(user_personas)} user personas and {len(conversations)} conversations")

    dataset = []
    for idx, (user_persona, conversation) in enumerate(tqdm(zip(user_personas, conversations), total=len(user_personas), desc="Ground truth scores")):
        tqdm.write(f"Scoring: {user_persona.name} with conversation {conversation.user_id}")
        
        def get_score():
            # Re-generating the grading rubric for each conversation to increase diversity of selected rubrics
            grading_rubric = CreateGradingRubricQuery(
                model_provider=openai_model_provider,
                model_id=args.rubric_generator_model_id,
                assistant=assistant
            ).query()

            score = GroundTruthJudgeQuery(
                model_provider=openai_model_provider,
                model_id=args.ground_truth_judge_model_id,
                grading_rubric=grading_rubric,
                assistant=assistant,
                conversation=conversation,
                user_persona=user_persona
            ).query()
            
            return score, grading_rubric
        
        with ThreadPoolExecutor(max_workers=args.num_attempts_per_conversation) as executor:
            futures = [executor.submit(get_score) for _ in range(args.num_attempts_per_conversation)]
            results = [future.result() for future in as_completed(futures)]
        
        scores = [result[0] for result in results]
        grading_rubrics = [result[1] for result in results]
        
        avg_score = int(round(sum(scores) / len(scores)))
        tqdm.write(f"Scores: {scores} -> Averaged: {avg_score}")

        dataset.append({
            "messages": [
                {"role": "user", "content": grading_rubric_template_prompt}
            ],
            "assistant_name": assistant.name,
            "assistant_description": assistant.description,
            "conversation_str": json.dumps(conversation.prompt_format, indent=4),
            "expected_judge_score": avg_score,
            "grading_rubric": grading_rubrics[0]  # Use a grading rubric from one of the attempts as a reference, won't actually be used for training
        })

    # Shuffle dataset with fixed seed for reproducibility
    random.seed(42)
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    
    # Split dataset into training and validation
    split_index = int(len(shuffled_dataset) * (1- args.validation_fraction))
    training_dataset = shuffled_dataset[:split_index]
    validation_dataset = shuffled_dataset[split_index:]
    
    print(f"Dataset split: {len(training_dataset)} training samples, {len(validation_dataset)} validation samples")
    
    # Helper function to upload a JSONL dataset to OpenAI without writing to disk
    def upload_dataset_to_openai(dataset, filename):
        """Uploads the given dataset (list of dicts) to OpenAI as a JSONL file."""
        # Convert dataset to newline-delimited JSON string
        jsonl_str = "\n".join(json.dumps(item) for item in dataset)

        # Create a bytes buffer with a name attribute so the OpenAI client
        # treats it like a file.
        bytes_buffer = io.BytesIO(jsonl_str.encode("utf-8"))
        bytes_buffer.name = filename  # type: ignore

        print(f"Uploading {filename} (size: {bytes_buffer.getbuffer().nbytes} bytes) to OpenAI…")

        response = openai_client.files.create(
            file=bytes_buffer,
            purpose="fine-tune",
        )

        print(f"Uploaded {filename}. File ID: {response.id}")
        return response.id

    # Upload training and validation datasets
    training_file_id = upload_dataset_to_openai(training_dataset, "training_dataset.jsonl")
    validation_file_id = upload_dataset_to_openai(validation_dataset, "validation_dataset.jsonl")

    print(f"Upload completed: Training file ID: {training_file_id}, Validation file ID: {validation_file_id}")

    try:
        result = create_fine_tuning_job(training_file_id, validation_file_id)
        print("Fine-tuning job created successfully!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Failed to create fine-tuning job: {e}")
            

