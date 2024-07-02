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
from CR_mod_reasoning_litellm_ReasoningChainCheck import CR_modified

# Load API keys
load_dotenv(os.path.join(config_dir, ".env"))
apikey = os.getenv('OPENAI_API_KEY')
##########################################

def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove the file.
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)




def get_parser():

    parser = argparse.ArgumentParser(description="Math Reasoning Agent")

    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--num_workers', type=int, default=8, help='number of processes')
    parser.add_argument('--answertrycnt', type=int, choices=range(0, 101), default=1, help='numbers of tries to answer')
    parser.add_argument('--problems_per_class', type=int, choices=range(0, 20), default=15, help='number of problems to solve per class')
    parser.add_argument('--model', type=str, choices=['gpt-3.5-turbo-16k', 'gpt4-1106-preview'], default='gpt-3.5-turbo-16k', help='model to use')
    parser.add_argument('--base_url', type=str, default='https://drchat.xyz', help='model to use')

    return parser




def worker(shared_data, counter_lock, blacklist_lock, pid: int, start_dif: int, appendToOriginal: bool, args):

    old_stdout = sys.stdout
    sys.stdout = open(os.path.join(thread_dir, f'PID_output_{pid}.txt'), 'w' if not appendToOriginal else 'a', encoding='utf-8')

    print(f"starting process: Process {pid}", flush=True)
    print(f"Try Count: {args.answertrycnt}", flush=True)
    paths_list = shared_data['paths_list']

    for i in range(start_dif, len(paths_list), args.num_workers):

        print(f"Solving Problem: {paths_list[i]}", flush=True)

        math_agent = CR_modified("gpt4-1106-preview")
        judge = Judger("gpt4-1106-preview")

        data = {}
        with open(paths_list[i], 'r') as f:
            data = json.load(f)
        
        lvl = data['level'][len(data['level']) - 1]
        
        tries = 0
        while (tries < args.answertrycnt):
            try:
                
                print("############# SOLUTION ############", flush=True)
                solution = math_agent.solve(data['type'], data['problem'], 0.0)
                print(solution, flush=True)
                
                print("############# JUDGEMENT ############", flush=True)
                judgement = judge.compare(data['problem'], data['type'], solution, data['solution'])
                print(judgement, flush=True)

                # Critical section.
                # increment needed values, dump json file.
                with counter_lock:
                    
                    print(f"PID: {pid} | In Lock")
                    shared_data['num_solved'].value += 1
                    if judgement == "Correct":
                        shared_data['num_correct'].value += 1

                    jsonDump = {"PID": f"{pid}",
                                "Number Solved": f"{shared_data['num_solved'].value}/{shared_data['total_num_problems']}",
                                "Running Accuracy": "{}".format(round(shared_data['num_correct'].value / shared_data['num_solved'].value, 3)),
                                "Level": data["level"],
                                "Correctness": judgement,
                                "Question Type": data['type'],
                                "Generated Solution": solution,
                                "Actual Solution": data["solution"],
                                "Question": data['problem']}
                
                    print(f"File Write ----------------------------------------------------------------------- PID: {pid}")
                    with open(shared_data['resultsFilePath'], 'a', encoding='utf-8') as file:
                        json.dump(jsonDump, file)
                        file.write("\n")
                        file.flush()

                    print(f"PID: {pid} | Out Lock", flush=True)
                
                break

            except Exception as e:

                print(f"PID: {pid} | | Exception Print: " + str(e), flush=True)
                tries += 1
        
        if tries >= args.answertrycnt:
            with blacklist_lock:
                path_dump = {"filepath" : paths_list[i]}
                with open(shared_data['blackListPath'], 'a', encoding='utf-8') as file:
                    json.dump(path_dump, file)
                    file.write("\n")
                    file.flush()




def main(args):

    resultsFileName = input("Type name of results file\n")
    print("The file you have specified may be unfinished by standards set in the arguments (problems_per_class).")
    appendToOriginal = input("Continue evaluating for specified file if unfinished? (Y | N)\n")
    appendToOriginal = True if "Y" else False

    if not appendToOriginal:
        clear_folder(thread_dir)

    resultsFilePath = os.path.join(main_dir, "results", f"{resultsFileName}.jsonl")

    if not os.path.exists(resultsFilePath):
        with open(resultsFilePath, 'a'):
            os.utime(resultsFilePath, None)

    if not appendToOriginal:
        os.remove(resultsFilePath)
        with open(resultsFilePath, 'a'):
            os.utime(resultsFilePath, None)
    else:
        with open(resultsFilePath, 'a'):
            os.utime(resultsFilePath, None)

    num_correct = 0
    num_solved = 0
    paths_list = []
    typeToFolder = {"Algebra": "algebra", "Counting & Probability" : "counting_and_probability", "Geometry" : "geometry", "Intermediate Algebra" : "intermediate_algebra", "Number Theory" : "number_theory", "Prealgebra" : "prealgebra", "Precalculus" : "precalculus"}
    problem_count_dictionary = {(folder + f"|{i}"): args.problems_per_class for i in range(1, 6) for folder in os.listdir(os.path.join(data_dir, "MATH", "test"))}

    if appendToOriginal:

        finished_questions_list = []
        blacklisted_questions_list = []

        if os.path.exists(os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl")):
            with open(os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl"), 'a') as json_file:
                os.utime(os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl"), None)
            
            with open(os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl"), 'r') as json_file:
                if os.path.getsize(os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl")) != 0:
                    for line in json_file:
                        q = json.loads(line)
                        blacklisted_questions_list.append(q["filepath"])
        else:
            with open(os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl"), 'a') as json_file:
                os.utime(os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl"), None)


        with open(resultsFilePath, 'r') as json_file:
            for line in json_file:
                test_data = json.loads(line)

                if int(test_data["Number Solved"].split("/")[0]) > num_solved:
                    num_solved = int(test_data["Number Solved"].split("/")[0])
                
                if test_data["Correctness"] == "Correct":
                    num_correct += 1

                qtype = typeToFolder[test_data["Question Type"]]
                qlevel = test_data["Level"][len(test_data["Level"]) - 1]

                problem_count_dictionary[f"{qtype}|{qlevel}"] -= 1
                finished_questions_list.append(test_data["Question"])
        
        
        for folder in os.listdir(os.path.join(data_dir, "MATH", "test")):
            for i in range(1, 6):

                if problem_count_dictionary[f"{folder}|{i}"] > 0:
                    aggregate_for_section = []
                    for file in os.listdir(os.path.join(data_dir, "MATH", "test", folder)):
                        data = {}
                        with open(os.path.join(data_dir, "MATH", "test", folder, file), 'r') as f:
                            data = json.load(f)

                        if (not data["problem"] in finished_questions_list) and (data["level"] == f"Level {i}") and (not os.path.join(data_dir, "MATH", "test", folder, file) in blacklisted_questions_list):
                            aggregate_for_section.append(os.path.join(data_dir, "MATH", "test", folder, file))
                    
                    number_to_choose = problem_count_dictionary[f"{folder}|{i}"]

                    for k in range(number_to_choose):
                        rand_idx = random.randint(0, len(aggregate_for_section) - 1)
                        paths_list.append(aggregate_for_section[rand_idx])
                        aggregate_for_section.pop(rand_idx)

    else:
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

    
    
    print(num_solved)
    print(num_correct)
    ##########################################
    # Shared resources between threads
    ##########################################
    # To keep track of counts on which problems have been solved.
    manager = Manager()
    shared_data = manager.dict()
    shared_data['num_solved'] = manager.Value('i', num_solved) # shared write
    shared_data['num_correct'] = manager.Value('i', num_correct) # shared write
    shared_data['total_num_problems'] = args.problems_per_class * 35 # shared access
    shared_data['resultsFilePath'] = resultsFilePath # shared write
    shared_data['blackListPath'] = os.path.join(blacklist_dir, f"{resultsFileName}_BlackListQuestions.jsonl") # shared write
    shared_data['paths_list'] = paths_list  # shared access

    # Manage Shared access
    counter_lock = Semaphore(1)
    blacklist_lock = Semaphore(1)
    #########################################

    processes = [Process(target=worker, args=(shared_data, counter_lock, blacklist_lock, i, i, appendToOriginal, args)) for i in range(args.num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    



if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)