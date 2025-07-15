import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import io
import os
import random
from tqdm import tqdm
import requests

from openai import OpenAI

from data_models.assistant import load_assistant
from data_models.character_card import load_user_personas
from data_models.conversation import load_conversations
from llm_queries.llm_query import OpenAIModelProvider
from llm_queries.ground_truth_judge_query import GroundTruthJudgeQuery
from llm_queries.create_grading_rubric_query import CreateGradingRubricQuery

openai_client = OpenAI()
openai_model_provider = OpenAIModelProvider(OpenAI())


def create_fine_tuning_job(base_model_id, training_file_id, validation_file_id):
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
        "model": base_model_id,
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
                    "n_epochs": 2,
                    "learning_rate_multiplier": 2.0
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
    parser.add_argument("--assistant-definition-file", type=str, required=True,
                        help="Path to the file containing your assistant definition (name and description)")
    parser.add_argument("--user-personas-file", type=str, required=True,
                        help="Path to the file containing your generated user personas")
    parser.add_argument("--conversations-file", type=str, required=True,
                        help="Path to the file containing conversations between the generated personas and your assistant")
    parser.add_argument("--num-attempts-per-conversation", type=int, default=5,
                        help="Number of ground truth scores to generate per conversation. Scores are averaged to calculate a single ground truth score per conversation (default: 5)")
    parser.add_argument("--ground-truth-judge-model-id", type=str, default="o3",
                        help="Model ID for executing the ground truth grader judge (default: o3)")
    parser.add_argument("--judge-prompt-generator-model-id", type=str, default="o4-mini-2025-04-16",
                        help="Model ID for generating candidate LLM-as-a-judge prompts during fine-tuning (default: o4-mini-2025-04-16)")
    parser.add_argument("--validation-fraction", type=float, default=0.25,
                        help="Fraction of the dataset to withhold for validation during training (default: 0.25)")
    parser.add_argument("--validation-file-id", type=str, default=None,
                        help="Optional: Use an existing OpenAI file ID for validation data instead of uploading new data")
    parser.add_argument("--training-file-id", type=str, default=None,
                        help="Optional: Use an existing OpenAI file ID for training data instead of uploading new data")
    args = parser.parse_args()
    
    # Load assistant definition, user personas, and conversations
    assistant = load_assistant(args.assistant_definition_file)
    user_personas = load_user_personas(args.user_personas_file)
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
                model_id=args.judge_prompt_generator_model_id,
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
    split_index = int(round(len(shuffled_dataset) * (1 - args.validation_fraction)))
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
    training_file_id = args.training_file_id if args.training_file_id else upload_dataset_to_openai(training_dataset, "training_dataset.jsonl")
    validation_file_id = args.validation_file_id if args.validation_file_id else upload_dataset_to_openai(validation_dataset, "validation_dataset.jsonl")

    print(f"Upload completed: Training file ID: {training_file_id}, Validation file ID: {validation_file_id}")

    try:
        result = create_fine_tuning_job(args.judge_prompt_generator_model_id, training_file_id, validation_file_id)
        print("Fine-tuning job created successfully!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Failed to create fine-tuning job: {e}")
            

