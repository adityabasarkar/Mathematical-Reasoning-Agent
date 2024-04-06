###########
 # SETUP #
###########
from time import sleep
import os
from dotenv import load_dotenv
import sys

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
from reasoning_agent import MathReasoning

# Models to be used
gpt4 = guidance.models.OpenAIChat(model="gpt4-1106-preview", tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url="https://drchat.xyz")
lm = guidance.models.OpenAIChat(model="gpt-3.5-turbo-16k", tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url="https://drchat.xyz")

def get_parser():
    parser = argparse.ArgumentParser(description="Math Reasoning Agent")
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--answertrycnt', type=int, choices=range(0, 101), default=4, help='numbers of tries to answer')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-16k', help='model to use')
    return parser

parser = get_parser()
args = parser.parse_args()

def main():
    # To keep track of counts on which problems have been solved.
    problems_per_class = 15
    problem_count_dictionary = {}
    for folder in os.listdir(os.path.join(data_dir, "MATH", "test")):
        problem_count_dictionary[folder + "|1"] = 15
        problem_count_dictionary[folder + "|2"] = 15
        problem_count_dictionary[folder + "|3"] = 15
        problem_count_dictionary[folder + "|4"] = 15
        problem_count_dictionary[folder + "|5"] = 15

if __name__=="__main__":
    main()