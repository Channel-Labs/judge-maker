import argparse
import json
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

from openai import OpenAI

from data_models.assistant import Assistant, load_assistant
from data_models.conversation import Conversation, Message, ROLE
from llm_queries.create_grading_rubric_query import CreateGradingRubricQuery
from llm_queries.judge_conversation_query import JudgeConversationQuery
from llm_queries.llm_query import OpenAIModelProvider

openai_model_provider = OpenAIModelProvider(OpenAI())


class ChatbotArenaWinnerPredictor:
  def __init__(self, assistant: Assistant, judge_model_id: str, judge_prompt_generator_model_id: str):
    self.assistant = assistant
    self.judge_model_id = judge_model_id
    self.judge_prompt_generator_model_id = judge_prompt_generator_model_id

  @staticmethod
  def _format_conversation(prompt_messages, response_messages) -> Conversation:    
      messages = []
      for i, msg in enumerate(prompt_messages):
          messages.append(Message(role=ROLE.user, content=msg, timestamp=0, message_id=2*i))
          messages.append(Message(role=ROLE.assistant, content=response_messages[i], timestamp=0, message_id=2*i+1))
      
      return Conversation(id=0, user_id=0, messages=messages)


  def calculate_score(self, prompt_messages, response_messages, grading_rubric):
  
    conversation = self._format_conversation(prompt_messages, response_messages)

    return JudgeConversationQuery(
      model_provider=openai_model_provider,
      model_id=self.judge_model_id,
      grading_rubric=grading_rubric,
      assistant=self.assistant,
      conversation=conversation
    ).query()

  def determine_winner(self, prompt_messages, response_a_messages, response_b_messages):
    grading_rubric = CreateGradingRubricQuery(
      model_provider=openai_model_provider,
      model_id=self.judge_prompt_generator_model_id,
      assistant=self.assistant
    ).query()

    score_a = self.calculate_score(prompt_messages, response_a_messages, grading_rubric)
    score_b = self.calculate_score(prompt_messages, response_b_messages, grading_rubric)

    if score_a > score_b:
      return "a"
    else:
      return "b"

  
def process_row(row, predictor: ChatbotArenaWinnerPredictor):
    prompt_messages = json.loads(row['prompt'])
    response_a_messages = json.loads(row['response_a'])
    response_b_messages = json.loads(row['response_b'])
    
    return predictor.determine_winner(prompt_messages, response_a_messages, response_b_messages)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--assistant-definition-file", type=str, required=True, 
                      help="Path to the file containing your assistant definition (name and description)")
  parser.add_argument("--evaluation-conversations-file", type=str, default="data/evaluation/chatbot_arena_multi_turn_conversations.csv",
                      help="Path to the file containing conversations to evaluate")
  parser.add_argument("--output-file", type=str, default="predicted_chatbot_arena_winners.csv",
                      help="Path where evaluation results will be saved")
  parser.add_argument("--num-conversations", type=int, default=200,
                      help="Number of conversations to judge (default: 200)")
  parser.add_argument("--judge-prompt-generator-model-id", type=str, required=True,
                      help="Model ID for generating judge prompts. Use your fine-tuned model ID to evaluate fine-tuning performance, or a base model ID to evaluate the baseline")
  parser.add_argument("--judge-model-id", type=str, default="o3",
                      help="Model ID for judging conversations in the evaluation file (default: o3)")
  args = parser.parse_args()

  assistant = load_assistant(args.assistant_definition_file)
  df = pd.read_csv(args.evaluation_conversations_file)[:args.num_conversations]
  actual_winners = list()
  for i, row in df.iterrows():
    if row['winner_model_a'] == 1:
      actual_winners.append("a")
    else:
      actual_winners.append("b")

  predictor = ChatbotArenaWinnerPredictor(assistant, args.judge_model_id, args.judge_prompt_generator_model_id)

  predicted_winners = list()
  with ThreadPoolExecutor(max_workers=6) as executor:
      # Submit all tasks
      future_to_index = {executor.submit(process_row, row, predictor): i for i, row in df.iterrows()}
      
      # Initialize results list with None values to maintain order
      results = [None] * len(df)
      
      # Collect results as they complete
      for future in tqdm.tqdm(as_completed(future_to_index), total=len(df)):
          index = future_to_index[future]
          try:
              winner = future.result()
              results[index] = winner
          except Exception as exc:
              print(f'Row {index} generated an exception: {exc}')
              results[index] = None  # or handle error as appropriate
      
      # Replace None results with "error" to maintain equal lengths of predicted and actual winners
      predicted_winners = [result if result is not None else "error" for result in results]

  print(f"Predicted winners length: {len(predicted_winners)}, Actual winners length: {len(actual_winners)}")
  print("Num correct: ", sum([predicted_winners[i] == actual_winners[i] for i in range(len(predicted_winners))]))

  df['predicted_winner'] = predicted_winners
  df['actual_winner'] = actual_winners
  df.to_csv(args.output_file, index=False)