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
solution, solution_dict = math_agent.solve("Algebra", "If you continue this pattern in which each line-segment extremity is replaced by a gradually smaller Y in the next figure, in the manner shown, how many endpoints will Figure 5 have?\n\n[asy]\ndraw((0,0)--(0,-3),linewidth(.75));\ndraw((0,0)--(-2,2),linewidth(.75));\ndraw((0,0)--(2,2),linewidth(.75));\nlabel(\"Figure 1\",(0,-3),S);\n\ndraw((5,0)--(5,-2),linewidth(.75));\ndraw((4,-3)--(5,-2)--(6,-3),linewidth(.75));\ndraw((4,1)--(5,0)--(6,1),linewidth(.75));\ndraw((3,1)--(4,1)--(4,2),linewidth(.75));\ndraw((6,2)--(6,1)--(7,1),linewidth(.75));\nlabel(\"Figure 2\",(5,-3),S);\n\ndraw((10,0)--(10,-2),linewidth(.75));\ndraw((9.5,-2.5)--(10,-2)--(10.5,-2.5),linewidth(.75));\ndraw((9,-2.5)--(9.5,-2.5)--(9.5,-3),linewidth(.75));\ndraw((11,-2.5)--(10.5,-2.5)--(10.5,-3),linewidth(.75));\n\ndraw((9,1)--(10,0)--(11,1),linewidth(.75));\ndraw((8.5,1)--(9,1)--(9,1.5),linewidth(.75));\ndraw((11.5,1)--(11,1)--(11,1.5),linewidth(.75));\ndraw((8.25,.75)--(8.5,1)--(8.25,1.25),linewidth(.75));\ndraw((8.75,1.75)--(9,1.5)--(9.25,1.75),linewidth(.75));\ndraw((10.75,1.75)--(11,1.5)--(11.25,1.75),linewidth(.75));\ndraw((11.75,1.25)--(11.5,1)--(11.75,.75),linewidth(.75));\nlabel(\"Figure 3\",(10,-3),S);\n\n[/asy]", 0.0)
print(math_agent.language_model._state)
