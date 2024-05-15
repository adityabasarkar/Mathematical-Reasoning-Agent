###########
from time import sleep
import os
from dotenv import load_dotenv
import sys
import json
from io import StringIO
import re
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

import argparse
# from reasoning_agent import CRPoweredSelfDiscover, Judger

from time import sleep
import litellm
from litellm import completion
litellm.set_verbose = False
import guidance
from guidance import models, gen
from guidance import select
from guidance import user, assistant, system
import tiktoken
from tiktoken import encoding_name_for_model
import multiprocessing


def run_code(code, output_queue, error_queue):
    result_out = StringIO()
    sys.stdout = result_out
    try:
        exec(code, globals())
        output_queue.put(result_out.getvalue())
    except Exception as e:
        output_queue.put(result_out.getvalue())
        error_queue.put(e)
    finally:
        sys.stdout = sys.__stdout__


def run_code_with_timeout(code, timeout):
    output_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()
    # Create a process that runs the run_code function
    process = multiprocessing.Process(target=run_code, args=(code, output_queue, error_queue))
    process.start()  # Start the process
    process.join(timeout)  # Allow the process to run for 'timeout' seconds

    output = None
    error = None

    if process.is_alive():  # If the process is still running after the timeout
        process.terminate()  # Terminate the process
        process.join()  # Wait for the process to clean up
        output = output_queue.get() if not output_queue.empty() else ""
        error = Exception("The code has been terminated due to timeout. The code has run for longer than the allowed time.")
        return output, error
    
    output = output_queue.get() if not output_queue.empty() else ""
    error = error_queue.get() if not error_queue.empty() else None

    return output, error

class InvalidOutputException(Exception):
    def __init__(self, message):
        super().__init__(message)
    
class CR_modified:

    def __init__(self, lm: str):
        self.language_model = lm

    def solve(self, question_type: str, question: str, temperature: float):

        print("Generating important information", flush=True)
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        Your role is to extract important information from problem in the form of "initial propositions"
                        ""","role": "user"},
                        { "content": f"""
                        Problem to solve: {question}\n
                        Question type: {question_type}\n

                        Generate a series initial propositions to gather context for solving the problem.

                        The propositions should follow the following instructions:\n
                        1. Each initial proposition only provides context (Does not actively solve the problem), contains important info to solve the problem, follows format: [Proposition #] Proposition text\n
                        2. Any number of initial propositions.\n
                        3. Minimize total propositions to under 10.\n
                        4. Propositions separated by \"|\" \n\n

                        Example Proposition Format:\n
                        [1] Initial Proposition 1 | [2] Initial Proposition 2
                        Minimize token usage in output for lower costs.
                        ""","role": "user"}],
            temperature=temperature
        )

        important_info = response.choices[0].message.content

        # Generate the program
        print("Generating program / subsequent propositions", flush=True)
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        Given a problem and extracted insights about the problem, your role is to solve the problem by proposing subsequent propositions (SPs) as a program.
                        ""","role": "user"},
                        { "content": f"""
                        Problem to solve: {question}\n
                        Question type: {question_type}\n
                        Initial Propositions: {important_info}\n

                        Generate a series of subsequent propositions (SPs) as a program, to solve the problem.

                        The propositions should follow the following instructions:\n
                        1. Python program contains SPs and corresponding program sections
                        2. Subsequent propositions are steps taken to actively solve problem. They specify which of the previous propositions they follow from (including initial propositions).\n
                        3. Follow format: [Proposition #] [based on initial/subsequent proposition #s] Proposition Text\n
                        4. The python program: a. Can only use numpy, math, sympy, and default python libraries, b. Defines all variables with numbers before use, c. Is brief and simple as possible, contains no recursion.\n
                        5. The python program should be able to run as quickly as possible (constant time preferred). There should be no infinite loops. 
                        6. Wrapped in <PROGRAM></PROGRAM> tags and outputs the answer to the problem explicitly USING print().\n
                        7. Maximize use of Python for accuracy, minimize total propositions to under 25.\n
                        8. Every program section after every subseqeuent proposition MUST PRINT any useful information using the print() function to aid in future debugging.
                        9. Propositions separated by \"|\" \n\n

                        Example Subsequent Proposition Format:\n
                        <PROGRAM>\n# [3] [1,2] Subsequent Proposition 3\nProgram for proposition 3\nprint() useful variables/info from proposition 3\n# [4] [1,2,3] Subsequent Proposition 4\nProgram for proposition 4\nprint() useful variables/info from proposition 4</PROGRAM> 
                        Minimize token usage in output for lower costs. Keep the propositions and program minimal and succinct.
                        ""","role": "user"}],
            temperature=temperature
        )
        
        prgm = response.choices[0].message.content

        match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', prgm, re.DOTALL)
        tries = 0
        max_tries = 4
        output = None
        feedback_array = None
        current_error = None

        while (tries < max_tries):

            try:

                if (tries > 0):

                    match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', prgm, re.DOTALL)

                if match:
                    
                    print("Running program", flush=True)
                    code_to_run = match.group(1)
                    
                    output, error = run_code_with_timeout(code_to_run, 25)
                    
                    if error:
                        raise error

                    # Verify if the output is acceptable.
                    print("Verifying output", flush=True)
                    response = completion(
                        api_key=apikey,
                        base_url="https://drchat.xyz",
                        model = self.language_model,
                        custom_llm_provider="openai",
                        messages = [{ "content": f"""
                                    Your role is to determine if the answer in the program output is a valid answer to the problem. The program may have multiple print statements. Most are for debugging, but one of them should output the answer.
                                    ""","role": "user"},
                                    { "content": f"""
                                    Problem to solve: {question}\n
                                    Question type: {question_type}\n
                                    Important insights: {important_info}\n
                                    Program for solving problem: {prgm}\n
                                    Program output: {output}\n

                                    Evaluate whether a program's output is an acceptable answer to a question, focusing not on correctness but on 
                                    relevance and appropriateness given the question's context. For example, in a question about a circle's area, 
                                    while '2*pi' might not be the correct answer, it should be considered acceptable if it relates meaningfully to 
                                    the context provided.

                                    Part 1 of the response should be a short explanation. Part 2 should specify "Y" if acceptable, and "N" if not acceptable.\n
                                    Part 3 should exist if not acceptable (N). It details the issue with the program and suggests how the program can be improved for better output. The 3 parts of the response should be separated by "|||"
                                    
                                    It is imparative that your response strictly follows the following format since your output will be processed externally:\n
                                    Format: "[Explanation] ||| [Y or N] ||| [Program improvements and suggestions (if N)]"
                                    ""","role": "user"}],
                        temperature=temperature
                    )

                    feedback_array = response.choices[0].message.content.split("|||")

                    # If unacceptable, raise an error. 
                    if "N" in feedback_array[1]:
                        raise InvalidOutputException(f"The output of your program was deemed invalid. Here are some suggestions:\n{feedback_array[2]}")

                tries = 0
                current_error = None
                break

            except Exception as e:
                
                print("Program Failed", flush=True)
                current_error = str(e)
                tries += 1

                if (tries < max_tries):
                    
                    # Fix the program and keep the comments. 
                    print("Attempting program fix", flush=True)
                    response = completion(
                        api_key=apikey,
                        base_url="https://drchat.xyz",
                        model = self.language_model,
                        custom_llm_provider="openai",
                        messages = [{ "content": f"""
                                    The program created to answer the question did not produce the expected output. Your job is to consider the feedback / error, question, important insights, and output, and fix the program.
                                    ""","role": "user"},
                                    { "content": f"""
                                    Problem to solve: {question}\n
                                    Question type: {question_type}\n
                                    Important insights: {important_info}\n
                                    Program for solving problem: {prgm}\n
                                    Program output: {output}\n
                                    Feedback / Error: {current_error}\n

                                    Keep the comments the same but fix the issues in the program given the feedback.
                                    You response should be an explanation of what the problem is, why and where the error must have occured, and what should be changed followed by a python program encapsulated within the <PROGRAM></PROGRAM> brackets.
                                    Pay attention to the program output.
                                    The program should print out the answer to the problem exclusively using the print() statement
                                    Every program section after every subseqeuent proposition MUST PRINT any useful information using the print() function to aid in future debugging.
                                    The python program should be able to run as quickly as possible (constant time preferred). There should be no infinite loops.
                                    Minimize token usage in output for lower costs. Keep the propositions and program minimal and succinct.

                                    Program should follow the example format:\n
                                    <PROGRAM>\n# [3] [1,2] Subsequent Proposition 3\nProgram for proposition 3\nprint() useful variables/info in proposition 3\n# [4] [1,2,3] Subsequent Proposition 4\nProgram for proposition 4\nprint() useful variables/info in proposition 4</PROGRAM>
                                    ""","role": "user"}],
                        temperature=temperature
                    )

                    prgm = response.choices[0].message.content

                else:

                    break

        print("Generating final answer", flush=True)
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        Given a problem, initial propositions about the problem, a program to solve the problem, and its output, determine the answer to the problem. The output may or may not be useful and you may have to 
                        develop your own solution given the rest of the context if the output isn't useful for solving the problem.
                        ""","role": "user"},
                        { "content": f"""
                        Problem to solve: {question}\n
                        Question type: {question_type}\n
                        Initial Propositions: {important_info}\n
                        Program for solving problem: {prgm}\n
                        Program output: {output}\n
                        Error or Feedback: {current_error}\n

                        The program may or may not have errors and could have generated output only to a certain point in the program. The output may not have been valid and there could be feedback. The program could have timed out as well.
                        There is also the possibility of muliple solutions, and only one of them might be valid in the context of the question.
                        Example: In a question that relates the lengths of sides of two shapes, the equations that relate the two, might yield the following solutions: (-2, -4), (2, 3). Since -2 or -4 cannot
                        be the lengths of any shape, the answer would be within the solution (2, 3). The answer could be 2 or 3 depending on the context.
                        Make sure to consider possibilities like the above.
                        Look at the output of the program as well as the program itself, and the initial propositions, and determine what the answer to the problem is. Box the answer using
                        \\boxed.\n\n

                        Minimize token usage in output for lower costs. Keep the explanations brief and succinct.
                        ""","role": "user"}],
            temperature=temperature
        )
        
        answer = response.choices[0].message.content

        return answer