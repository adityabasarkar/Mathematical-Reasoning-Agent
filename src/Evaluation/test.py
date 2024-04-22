###########
 # SETUP #
###########
from time import sleep
import os
from dotenv import load_dotenv
import sys
import json
# Main directory
main_dir = os.path.abspath(__file__)
for i in range(0, 3):
    main_dir = os.path.dirname(main_dir)

# Configuration directory
config_dir = os.path.join(main_dir, "src", "config_files")

# Reasoning agent directory
reasoning_agent_dir = os.path.join(main_dir, "src", "Reasoning Agent")

# Dataset directory
data_dir = os.path.join(main_dir, "Datasets")

# Add paths to configuration
sys.path.append(main_dir)
sys.path.append(config_dir)
sys.path.append(reasoning_agent_dir)
sys.path.append(data_dir)

# Load API keys
load_dotenv(os.path.join(config_dir, ".env"))
apikey = os.getenv('OPENAI_API_KEY')

import guidance
from guidance import models, gen
from guidance import select
from guidance import user, assistant, system
import tiktoken
from tiktoken import encoding_name_for_model
import argparse
from reasoning_agent import CRPoweredSelfDiscover, Judger


gpt4 = guidance.models.OpenAIChat(model="gpt4-1106-preview", tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url="https://drchat.xyz")
lm = guidance.models.OpenAIChat(model="gpt-3.5-turbo-16k", tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url="https://drchat.xyz")
math_agent = CRPoweredSelfDiscover(lm)
judge = Judger(gpt4)

data = {}
data_path = os.path.join(data_dir, "MATH", "test", "intermediate_algebra", "90.json")
with open(data_path, 'r') as f:
    data = json.load(f)
question = data["problem"]
question_type = data["type"]
actual_solution = data["solution"]


solution, solution_dict = math_agent.solve(question_type, question, 0.0)
j = judge.compare(question, question_type, solution, actual_solution)


print(solution)
print(j['correctness'])
print(math_agent.language_model.token_count)
print(judge.language_model.token_count)