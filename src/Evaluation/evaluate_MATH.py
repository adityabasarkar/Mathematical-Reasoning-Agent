###########
 # SETUP #
###########
from time import sleep
import os
from dotenv import load_dotenv
import sys
import json
from multiprocessing import Semaphore, Process, Manager
import guidance
from guidance import models, gen
from guidance import select
from guidance import user, assistant, system
import tiktoken
from tiktoken import encoding_name_for_model
import argparse

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

# Add paths to configuration
sys.path.append(main_dir)
sys.path.append(config_dir)
sys.path.append(reasoning_agent_dir)
sys.path.append(data_dir)
from reasoning_agent import CRPoweredSelfDiscover, Judger

# Load API keys
load_dotenv(os.path.join(config_dir, ".env"))
apikey = os.getenv('OPENAI_API_KEY')
##########################################


def get_parser():

    parser = argparse.ArgumentParser(description="Math Reasoning Agent")

    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--num_workers', type=int, default=8, help='number of processes')
    parser.add_argument('--answertrycnt', type=int, choices=range(0, 101), default=1, help='numbers of tries to answer')
    parser.add_argument('--problems_per_class', type=int, choices=range(0, 20), default=15, help='number of problems to solve per class')
    parser.add_argument('--model', type=str, choices=['gpt-3.5-turbo-16k', 'gpt4-1106-preview'], default='gpt-3.5-turbo-16k', help='model to use')
    parser.add_argument('--base_url', type=str, default='https://drchat.xyz', help='model to use')

    return parser




def worker(shared_data, counter_lock, pid: int, start_dif: int, args):

    print(f"starting process: Process {pid}")
    print(f"Try Count: {args.answertrycnt}")
    paths_list = shared_data['paths_list']

    for i in range(start_dif, len(paths_list), args.num_workers):
            
        gpt4 = guidance.models.OpenAIChat(model="gpt4-1106-preview", tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url=args.base_url)
        lm = guidance.models.OpenAIChat(model=args.model, tokenizer=tiktoken.get_encoding("cl100k_base"), api_key=apikey, caching=True, base_url=args.base_url)

        math_agent = CRPoweredSelfDiscover(lm)
        judge = Judger(gpt4)

        data = {}
        with open(paths_list[i], 'r') as f:
            data = json.load(f)
        
        lvl = data['level'][len(data['level']) - 1]
        
        tries = 0
        while (tries < args.answertrycnt):
            try:

                # Parallel Solving
                lm.reset()
                gpt4.reset()
                math_agent = CRPoweredSelfDiscover(lm)
                judge = Judger(gpt4)
                solution, solution_dict = math_agent.solve(data['type'], data['problem'], 0.0)
                judgement = judge.compare(data['problem'], data['type'], solution, data['solution'])
                
                print(solution)

                # Critical section.
                # increment needed values, dump json file.
                with counter_lock:
                    
                    print(f"PID: {pid} | In Lock")
                    shared_data['num_solved'].value += 1
                    if judgement["correctness"] == "Correct":
                        shared_data['num_correct'].value += 1

                    jsonDump = {"PID": f"{pid}",
                                "Number Solved": f"{shared_data['num_solved'].value}/{shared_data['total_num_problems']}",
                                "Running Accuracy": "{}".format(round(shared_data['num_correct'].value / shared_data['num_solved'].value, 3)),
                                "Level": data["level"],
                                "Correctness": judgement["correctness"],
                                "Token Count": math_agent.language_model.token_count,
                                "Question Type": data['type'],
                                "Generated Solution": solution,
                                "Actual Solution": data["solution"],
                                "Question": data['problem']}
                
                    print(f"File Write ----------------------------------------------------------------------- PID: {pid}")
                    with open(shared_data['resultsFilePath'], 'a') as file:
                        json.dump(jsonDump, file)
                        file.write("\n")
                        file.flush()

                    print(f"PID: {pid} | In Lock")
                
                break

            except Exception as e:
                print(f"PID: {pid} | | Exception Print: " + str(e))
                tries += 1
            




def main(args):

    resultsFileName = input("Type name of results file\n")
    resultsFilePath = os.path.join(main_dir, "results", f"{resultsFileName}.jsonl")

    if not os.path.exists(resultsFilePath):
        with open(resultsFilePath, 'a'):
            os.utime(resultsFilePath, None)
    else:
        os.remove(resultsFilePath)
        with open(resultsFilePath, 'a'):
            os.utime(resultsFilePath, None)

    paths_list = []
    problem_count_dictionary = {(folder + f"|{i}"): args.problems_per_class for i in range(1, 6) for folder in os.listdir(os.path.join(data_dir, "MATH", "test"))}

    for i, folder in enumerate(os.listdir(os.path.join(data_dir, "MATH", "test"))):
        for lvl in range(1, 6):
            for file in os.listdir(os.path.join(data_dir, "MATH", "test", folder)):
                data = {}
                with open(os.path.join(data_dir, "MATH", "test", folder, file), 'r') as f:
                    data = json.load(f)
                level = data['level'][len(data['level']) - 1]
                if (lvl == int(level)):
                    paths_list.append(os.path.join(data_dir, "MATH", "test", folder, file))
                    problem_count_dictionary[f"{folder}|{lvl}"] -= 1

                if (problem_count_dictionary[f"{folder}|{lvl}"] <= 0):
                    break

    

    ##########################################
    # Shared resources between threads
    ##########################################
    # To keep track of counts on which problems have been solved.
    manager = Manager()
    shared_data = manager.dict()
    shared_data['num_solved'] = manager.Value('i', 0) # shared write
    shared_data['num_correct'] = manager.Value('i', 0) # shared write
    shared_data['total_num_problems'] = args.problems_per_class * 35 # shared access
    shared_data['resultsFilePath'] = resultsFilePath # shared write
    shared_data['paths_list'] = paths_list  # shared access

    # Manage Shared access
    counter_lock = Semaphore(1)
    #########################################

    processes = [Process(target=worker, args=(shared_data, counter_lock, i, i, args)) for i in range(args.num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    



if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)