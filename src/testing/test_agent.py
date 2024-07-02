###########
 # SETUP #
###########
from time import sleep
import os
from dotenv import load_dotenv
import sys
import json
from multiprocessing import Semaphore, Process, Manager
import argparse
import shutil
import random

##################################
# Configuration and directories
##################################
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

# Thread Log directory
thread_dir = os.path.join(main_dir, "Thread Logs")

# Black List directory
blacklist_dir = os.path.join(main_dir, "Problem_Blacklist")

# Add paths to configuration
sys.path.append(main_dir)
sys.path.append(config_dir)
sys.path.append(reasoning_agent_dir)
sys.path.append(data_dir)

from CRSD_reasoning_agent_litellm import Judger
from CR_mod_reasoning_litellm_ConsistencyCheck import CR_modified

# Load API keys
load_dotenv(os.path.join(config_dir, ".env"))
apikey = os.getenv('OPENAI_API_KEY')
##########################################

math_agent = CR_modified("gpt4-1106-preview")
judge = Judger("gpt4-1106-preview")

question_type = input("Enter Question Type: ")
question = input("Enter Question: ")
actual_solution = input("Enter Actual Solution: ")
solution = math_agent.solve(question_type, question, 0.0)
judgement = judge.compare(question, question_type, solution, actual_solution)

print(solution)
print(judgement)