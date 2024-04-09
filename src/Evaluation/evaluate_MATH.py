###########
 # SETUP #
###########
from time import sleep
import os
from dotenv import load_dotenv
import sys
import json
from threading import Thread, Semaphore

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

def get_parser():

    parser = argparse.ArgumentParser(description="Math Reasoning Agent")

    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--answertrycnt', type=int, choices=range(0, 101), default=2, help='numbers of tries to answer')
    parser.add_argument('--problems_per_class', type=int, choices=range(0, 20), default=15, help='number of problems to solve per class')
    parser.add_argument('--model', type=str, choices=['gpt-3.5-turbo-16k', 'gpt4-1106-preview'], default='gpt-3.5-turbo-16k', help='model to use')
    parser.add_argument('--base_url', type=str, default='https://drchat.xyz', help='model to use')

    return parser

parser = get_parser()
args = parser.parse_args()




# Models to be used

# Shared resources between threads
# To keep track of counts on which problems have been solved.
problems_per_class = args.problems_per_class
problem_count_dictionary = {}
for folder in os.listdir(os.path.join(data_dir, "MATH", "test")):
    problem_count_dictionary[folder + "|1"] = problems_per_class
    problem_count_dictionary[folder + "|2"] = problems_per_class
    problem_count_dictionary[folder + "|3"] = problems_per_class
    problem_count_dictionary[folder + "|4"] = problems_per_class
    problem_count_dictionary[folder + "|5"] = problems_per_class
# Keep track of number correct
num_correct = 0
total_num_problems = args.problems_per_class * 35

pcd_lock = Semaphore(1)

def worker(folder: str, start_dif: int):
    
    folder_dir = os.path.join(data_dir, "MATH", "test", folder)
    files = [os.path.join(folder_dir, file) for file in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file))]

    gpt4 = guidance.models.OpenAIChat(model="gpt4-1106-preview", tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url=args.base_url)
    lm = guidance.models.OpenAIChat(model=args.model, tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url=args.base_url)

    math_agent = CRPoweredSelfDiscover(lm)
    judge = Judger(gpt4)

    for i in range(start_dif, 6, 2):
        for j in range(0, len(files)):
            data = {}
            with open(files[j], 'r') as f:
                data = json.load(f)
            
            lvl = data['level'][len(data['level']) - 1]
            if (problem_count_dictionary["{}|{}".format(folder, lvl)] > 0):
                break
            
            if (int(lvl) != i):
                continue
            
            tries = 0
            while (tries < args.answertrycnt):
                try:
                    solution, solution_dict = math_agent.solve(data['type'], data['problem'], 0.0)
                    judgement = judge(data['problem'], data['type'], solution, data['solution'])
                except Exception as e:
                    tries += 1
            

def main():
    threads = []
    for folder in os.listdir(os.path.join(data_dir, "MATH", "test")):
        t1 = Thread(target=worker, args=(folder, 1))
        t2 = Thread(target=worker, args=(folder, 2))
        threads.append(t1)
        threads.append(t2)
        t1.start()
        t2.start()


if __name__=="__main__":
    main()