import argparse
import json
import os 
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

from openai import OpenAI

from data_models.assistant import Assistant
from data_models.conversation import Conversation, Message, ROLE
from llm_queries.create_grading_rubric_query import CreateGradingRubricQuery
from llm_queries.judge_conversation_query import JudgeConversationQuery
from llm_queries.llm_query import OpenAIModelProvider

openai_model_provider = OpenAIModelProvider(OpenAI())


class ChatbotArenaWinnerPredictor:
  def __init__(self, assistant: Assistant, model_id: str, rubric_generator_model_id: str):
    self.assistant = assistant
    self.model_id = model_id
    self.rubric_generator_model_id = rubric_generator_model_id

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
      model_id=self.model_id,
      grading_rubric=grading_rubric,
      assistant=self.assistant,
      conversation=conversation
    ).query()

  def determine_winner(self, prompt_messages, response_a_messages, response_b_messages):
    grading_rubric = CreateGradingRubricQuery(
      model_provider=openai_model_provider,
      model_id=self.rubric_generator_model_id,
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
  parser.add_argument("--assistant-name", type=str, required=True)
  parser.add_argument("--assistant-description", type=str, required=True)
  parser.add_argument("--chatbot-arena-conversations-file", type=str, default="data/evaluation/chatbot_arena_multi_turn_conversations.csv")
  parser.add_argument("--output-file", type=str, default="predicted_chatbot_arena_winners.csv")
  parser.add_argument("--num-conversations", type=int, default=200)
  parser.add_argument("--rubric-generator-model-id", type=str, required=True)
  parser.add_argument("--model-id", type=str, default="o3")
  args = parser.parse_args()

  assistant = Assistant(name=args.assistant_name, description=args.assistant_description)
  df = pd.read_csv(args.chatbot_arena_conversations_file)[:args.num_conversations]

  predictor = ChatbotArenaWinnerPredictor(assistant, args.model_id, args.rubric_generator_model_id)

  predicted_winners = list()
  # Use ThreadPoolExecutor with pool size of 4
  with ThreadPoolExecutor(max_workers=4) as executor:
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

  actual_winners = list()
  for i, row in df.iterrows():
    if row['winner_model_a'] == 1:
      actual_winners.append("a")
    else:
      actual_winners.append("b")

  print(f"Predicted winners length: {len(predicted_winners)}, Actual winners length: {len(actual_winners)}")
  print("Num correct: ", sum([predicted_winners[i] == actual_winners[i] for i in range(len(predicted_winners))]))

  df['predicted_winner'] = predicted_winners
  df['actual_winner'] = actual_winners
  df.to_csv(args.output_file, index=False)