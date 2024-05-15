"""
Contains the main mathematical reasoning mechanism for the agent.
Uses guidance as the main method of manipulating LLMs
"""

###########
 # SETUP #
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
litellm.set_verbose=False
import guidance
from guidance import models, gen
from guidance import select
from guidance import user, assistant, system
import tiktoken
from tiktoken import encoding_name_for_model




class CRPoweredSelfDiscover:

    def __init__(self, lm: str):
        self.language_model = lm

        self.reasoning_modules = [
            "1. How could I devise an experiment to help solve that problem?",
            "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
            "3. How can I simplify the problem so that it is easier to solve?",
            "4. What are the key assumptions underlying this problem?",
            "5. What are the alternative perspectives or viewpoints on this problem?",
            "6. How can I break down this problem into smaller, more manageable parts?",
            "7. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
            "8. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
            "9. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
            "10. What is the core issue or problem that needs to be addressed?",
            "11. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
            "12. What are the potential obstacles or challenges that might arise in solving this problem?",
            "14. What indicators or metrics can be used?",
            "15. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
            "16. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
            "17. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
            "18. Does the problem require addressing systemic or structural issues rather than just individual instances?",
            "19. What kinds of solution typically are produced for this kind of problem specification?",
            "20. Let’s think step by step.",
            "21. Let’s make a step by step plan and implement it with good notation and explanation.",
            "22. What are the unknowns or variables in this problem?",
            "23. How can the equations in this problem be rearanged to make the problem easier to solve?",
            "24. What equations or formulas are relevant to this problem?",
            "25. Can I simplify the problem using basic algebraic principles (like combining like terms, factoring, or dividing)?",
            "26. Are there any constraints or conditions that must be considered?",
            "27. What kind of functions or equations am I dealing with and what are its properties?",
            "28. What assumptions am I making, and are they valid in this context?",
            "29. What is the most efficient strategy to solve this problem?",
            "30. What theorems or lemmas are applicable to this problem?",
            "31. Are there any symmetries or patterns in the number set?",
            "32. Are the events independent or dependent, and how does this affect the probability?",
            "33. What assumptions am I making about the randomness or fairness of the events?",
            "34. Can symmetry or geometric probability be used to solve this problem?"
        ]

    def solve(self, question_type: str, question: str, temperature: float):
        
        complete_solution_capsule = {}
        complete_solution_capsule["question"] = question
        complete_solution_capsule["question_type"] = question_type



        print("generate module selection reasoning")
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        Your task is to select the reasoning modules that are crucial to utilize in order to solve the following question: {question}.
                        For additional context, the question type for this question is {question_type}
                        
                        For the math question, you will choose reasoning modules and give a 1 sentence short explanation as to why the module is necessary.

                        Choose from the following modules:\n
                        Minimize token usage in output for lower costs.
                        ""","role": "user"}, 
                        { "content": "\n".join(self.reasoning_modules),"role": "user"}],
            temperature=0.5
        )
        module_selection_response = response.choices[0].message.content
        complete_solution_capsule["module_selection_response"] = module_selection_response



        print("generate adaptation")
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        These are your chosen modules and the reasoning for each one: {module_selection_response}\n
                        Now that you have chosen your reasoning modules, rephrase and specify 
                        each reasoning module so that it better helps in solving the given question: {question}\n
                        Minimize token usage in output for lower costs.
                        ""","role": "user"}],
            temperature=0.5
        )
        adaptation = response.choices[0].message.content
        complete_solution_capsule["adaptation"] = adaptation



        print("generate step by step plan")
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        Given the question: {question}, you have adapted the reasoning modules to your specific task. Now, operationalize the reasoning modules into a step-by-step reasoning plan to solve the problem.

                        This is your adaptation: {adaptation}\n

                        Instructions:\n
                        List the direct action steps / calculations needed to solve this mathematical problem, excluding verification, contextual understanding, or reflection.
                        List the steps in a numbered format, ensuring each one is essential for solving the problem and directly involved in the necessary computations or manipulations.\n
                        Minimize number of steps to <= 5.
                        Minimize token usage in output for lower costs.
                        ""","role": "user"}],
            temperature=0.0
        )
        implement_reason = response.choices[0].message.content
        complete_solution_capsule["implement_reason"] = implement_reason



        print("formalize steps")
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        Here is the reasoning structure you've generated before:\n
                        {implement_reason}\n

                        Given the set of steps you made, formalize them into the specific format below. DO NOT INCLUDE ANY OTHER TEXT OTHER THAN THE STEPS WITH THE SPECIFIED FORMAT.\n
                        Your output will be processed externally

                        Format:
                        [Step 1 || Step 2 || Step 3 || Step 4 || ...]
                        
                        Example Output:
                        [1. Identify the radius of the circle || 2. Identify the length of the side of the square || 3. Find the area of the square || 4. Find the area of the circle || 5. Sum the areas]
                        Minimize token usage in output for lower costs.
                        ""","role": "user"}],
            temperature=0.0
        )
        formal_steps = response.choices[0].message.content
        complete_solution_capsule["formal_steps"] = formal_steps
        print(formal_steps)

        original = formal_steps
        formatted_string = original[1:-1]
        steps_list = formatted_string.split("||")
        steps_list = [step.strip() for step in steps_list]

        # Now that we have generated the steps, we will use cumulative reasoning to solve each step.
        # Cumulative reasoning here should use the same idea as the paper, but doesn't use the same code as the
        # CR source code

        print("Solve problem with CR")
        complete_solution_capsule["steps_list"] = {}
        history_accum = []
        for i, step in enumerate(steps_list):
            # First try
            response = completion(
                api_key=apikey,
                base_url="https://drchat.xyz",
                model = self.language_model,
                custom_llm_provider="openai",
                messages = [{ "content": f"""
                            Given the specific problem, the CURRENT STEP, and additional context, you will carry out that step in a specific format that will be defined.
                            ""","role": "user"},
                            { "content": f"""
                            Problem to solve: {question}
                            Previous steps context: {history_accum}
                            Steps list: {steps_list}
                            CURRENT STEP: {steps_list[i]}\n

                            Generate a series initial propositions to gather context and subsequent propositions (as a program) to solve ONLY the CURRENT STEP. All information above is additional context.

                            The propositions should follow the following instructions:\n
                            1. Each initial proposition only provides context (Does not actively solve the step), contains important info to solve CURRENT STEP, follow format: [Proposition #] Proposition text\n
                            2. Any number of initial propositions.\n
                            3. After initial propositions, you will write a python program which contains the subsequent propositions and their corresponding program sections. 
                            3. Subsequent propositions actively solve the CURRENT STEP and specify which of the previous following propositions they follow from.\n
                            4. Subsequent propositions use context from any previous propositions, follow format: [Proposition #] [based on initial/subsequent proposition #s] Proposition Text\n
                            6. The python program: a. Uses sympy, numpy, math, and default python libraries, b. Defines all variables with numbers before use, c. Is brief and simple as possible, contains no recursion.\n
                            7. Programs are wrapped in <PROGRAM></PROGRAM> tags and output the answer to the current step explicitly USING print().\n
                            8. When solving systems of equations using sympy, solve for one variable at a time. For example, given two equations with the variables x and y, solve for x using the first equation, and plug that into the second to solve for y.
                            DO NOT try to do this: solutions = sp.solve((eq1, eq2, eq3, eq4), (x, y, z, s))
                            8. Maximize use of Python for accuracy, minimize total propositions to under 16.\n
                            9. Propositions separated by \"|\" \n\n

                            Example Proposition Format:\n
                            [1] Initial Proposition 1 | [2] Initial Proposition 2  <PROGRAM># [3] [1,2] Subsequent Proposition 4\nProgram for proposition 4\n# [4] [1,2,3] Subsequent Proposition 5\nProgram for proposition 5</PROGRAM> 
                            Minimize token usage in output for lower costs.
                            ""","role": "user"}],
                temperature=0.0
            )
            complete_solution_capsule["steps_list"]["Step {}".format(i + 1)] = {}
            complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["instruction"] = step
            complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"] = response.choices[0].message.content
            print(step)
            print(complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"])
            
            match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"], re.DOTALL)
            tries = 0
            max_tries = 3
            old_stdout = sys.stdout
            output = None
            while (tries < max_tries):
                try:
                    if (tries > 0):
                        match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"], re.DOTALL)
                    if match:
                        print("Running Program")
                        code_to_run = match.group(1)
                        result = StringIO()
                        sys.stdout = result
                        exec(code_to_run)
                        sys.stdout = old_stdout
                        output = result.getvalue()

                    tries = 0
                    break

                except Exception as e:
                    
                    print("Program Failed")
                    sys.stdout = old_stdout
                    print(str(e))
                    tries += 1

                    if (tries < max_tries):
                        
                        # Fix program
                        response = completion(
                            api_key=apikey,
                            base_url="https://drchat.xyz",
                            model = self.language_model,
                            custom_llm_provider="openai",
                            messages = [{ "content": f"""
                                        The program in the <PROGRAM></PROGRAM> tags failed to run. Your job is to consider the error, the additional context, and fix the program so it does run.
                                        ""","role": "user"},
                                        { "content": f"""
                                        Question you are working towards solving: {question}\n Complete List of Steps for solving Question: {steps_list}\n Context for steps that have been solved: {history_accum}\n CURRENT STEP: {steps_list[i]}\n

                                        You have been given a math problem to solve and you have generated a list of steps to solve that problem. You have been given the entire list of steps to solve the\n
                                        problem, however, you are only responsible for solving the CURRENT STEP. Below is your previous attempt at solving the CURRENT STEP. The program has failed, and it is your job to fix it.\n
                                        This was your attempt at solving the CURRENT STEP:\n
                                        {complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"]}\n\n
                                        The program in the <PROGRAM></PROGRAM> tags failed to run and threw the following error:\n
                                        {str(e)}\n
                                        Your job is to fix and rewrite your program in the same format with the tags <PROGRAM></PROGRAM>. Make sure every variable you use in your program can be backed by some numerical value elsewhere in your program. Otherwise,
                                        your program will FAIL. Your generated output should solely be a python program encapsulated in the <PROGRAM></PROGRAM> tags that fixes the errors from the previous attempt.
                                        Minimize token usage in output for lower costs.
                                        ""","role": "user"}],
                            temperature=0.0
                        )
                        complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"] = re.sub(r'<PROGRAM>.*?</PROGRAM>', response.choices[0].message.content, complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"])
                        print(complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"])

                    else:
                        break
            
            
            if tries > 0:

                # If all else failed, just do it normally.
                response = completion(
                    api_key=apikey,
                    base_url="https://drchat.xyz",
                    model = self.language_model,
                    custom_llm_provider="openai",
                    messages = [{ "content": f"""
                                Given the specific problem, the CURRENT STEP, and additional context, you will carry out that step in a specific format that will be defined.
                                ""","role": "user"},
                                { "content": f"""
                                Problem to solve: {question}\nPrevious steps context: {history_accum}\nSteps list: {steps_list}\nCURRENT STEP: {steps_list[i]}\n

                                Generate a series initial propositions to gather context and subsequent propositions to solve ONLY the CURRENT STEP. All information above is additional context.

                                The propositions should follow the following instructions:\n
                                1. Each initial proposition only provides context (Does not actively solve the step), contains important info to solve CURRENT STEP, follow format: [Proposition #] Proposition text\n
                                2. Any number of initial propositions.\n
                                3. Subsequent propositions actively solve the CURRENT STEP and specify which of the previous following propositions they follow from.\n
                                4. Subsequent propositions use context from any previous propositions, follow format: [Proposition #] [based on initial/subsequent proposition #s] Proposition Text\n
                                5. minimize total propositions to under 16.\n
                                6. Propositions separated by \"|\" \n\n

                                Example Proposition Format:\n
                                [1] Initial Proposition 1 | [2] Initial Proposition 2 | [3] [1,2] Subsequent Proposition 4 | [4] [1,2,3] Subsequent Proposition 5 | [5] [1,3,4] <PROGRAM>Your python program to figure out final answer for CURRENT STEP goes here</PROGRAM> 
                                Minimize token usage in output for lower costs.
                                ""","role": "user"}],
                    temperature=0.0
                )
                complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"] = response.choices[0].message.content
                print(complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"])

            # For formalizing the solution. 
            response = completion(
                api_key=apikey,
                base_url="https://drchat.xyz",
                model = self.language_model,
                custom_llm_provider="openai",
                messages = [{ "content": f"""
                            Given the specific problem, the current step, and the solution for the current step, you are instructed to generate the final answer for the step.
                            ""","role": "user"},
                            { "content": f"""
                            Question you are working towards solving: {question}\n
                            Complete List of Steps for solving Question: {steps_list}\n
                            Context for steps that have been solved: {history_accum}\n
                            CURRENT STEP: {steps_list[i]}\n

                            Here is the solution to the CURRENT STEP: {complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["propositions"]}
                            Here is the program output for the python program inside the <PROGRAM></PROGRAM> tags: {output}

                            You have been given the CURRENT STEP, as well as the solution to the current step. Given this information as well as the program output. Your job is to formally
                            record JUST THE SOLUTION TO THE CURRENT STEP in the given format:\n
                            Step [#]. [Solution to the step]\n
                            Example:\n
                            Step 4. The radius of the circle is 15.
                            Minimize token usage in output for lower costs. Make sure that the program output is mentioned in the solution record
                            ""","role": "user"}],
                temperature=0.0
            )

            complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["step_answer"] = response.choices[0].message.content
            current_step_solution = response.choices[0].message.content
            print(complete_solution_capsule["steps_list"]["Step {}".format(i + 1)]["step_answer"])
            history_accum.append(f"{step} || Answer to Step {i + 1}: {current_step_solution}")


        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        Given the specific question and the answers to all the steps for solving the question, your job is to formulate the final answer to the question. 
                        ""","role": "user"},
                        { "content": f"""
                        Question you are working towards solving: {question}\n
                        Context for steps that have been solved: {history_accum}\n

                        Now that you have solved all the steps of your reasoning plan. Use all the information to formulate your final answer for the the question.\n
                        Make sure to write your answer with the correct units and keep your answer as simple as possible for easy grading.
                        Minimize token usage in output for lower costs.
                        ""","role": "user"}],
            temperature=0.0
        )

        complete_solution_capsule["final_solution"] = response.choices[0].message.content
        print(complete_solution_capsule["final_solution"])

        print ("############################ ||END SOLUTION|| ############################")
        return complete_solution_capsule["final_solution"], complete_solution_capsule

class Judger:
    def __init__(self, lm: str) -> None:
        self.language_model = lm
        self.valid_correctness = ["Correct", "Wrong", "Unknown"]

    def compare(self, question_content: str, question_subject: str, final_answer: str, ground_truth: str):
        
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        The following is a ground truth answer to a math problem: {ground_truth}
                        State the answer that is boxed in the most succinct way possible (to reduce output cost) and include units (if any)
                        ""","role": "user"}],
            temperature=0.0
        )

        ground_truth_answer = response.choices[0].message.content

        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
                        Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct. 
                        ""","role": "user"},
                        { "content": f"""
                        Problem Subject: "{question_subject}", Problem Content: "{question_content}"

                        First, given the ground truth answer, just state the actual answer that is boxed. This is the actual answer that is extracted from the ground truth answer\n
                        Next, compare the final_answer and the ground_truth answer. Don't be strict on the format but check the content. Make sure to consider the context of the question when making your decision . 
                        Is the final_answer correct, given the ground truth answer (ground_truth_answer)? 
                        Think step by step and write a short explanation for whether the two answers match and if further clarification is needed.
                        For further clarification, write a python program enclosed in the <PROGRAM> </PROGRAM> tags. Output will be returned for use.
                        Example: If final_answer is 1.732, and ground_truth_answer is sqrt(3), correctness is uncertain, so write python program:
                        <PROGRAM>import math\nprint(str(math.sqrt(3)))</PROGRAM>
                        It is of utmost importance that you correctly analyze the two answers and make a correct judgement.\n\n
                        "final_answer": "{final_answer}", \n"ground_truth_answer": "{ground_truth_answer}"
                        ""","role": "user"}],
            temperature=0.0
        )

        compare_rationale = response.choices[0].message.content

        match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', compare_rationale, re.DOTALL)
        output = None

        if match:

            code_to_run = match.group(1)

            try:

                old_stdout = sys.stdout
                result = StringIO()
                sys.stdout = result
                exec(code_to_run)
                output = result.getvalue()
                sys.stdout = old_stdout

                response = completion(
                    api_key=apikey,
                    base_url="https://drchat.xyz",
                    model = self.language_model,
                    custom_llm_provider="openai",
                    messages = [{ "content": f"""
                                YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
                                Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct. 
                                ""","role": "user"},
                                { "content": f"""
                                Problem Subject: "{question_subject}", Problem Content: "{question_content}"

                                Here is the final answer and the ground truth answer:\n
                                "final_answer": "{final_answer}", \n"ground_truth_answer": "{ground_truth_answer}"

                                You have compared the final answer to the ground truth answer:\n
                                {compare_rationale}

                                The program you have written within the <PROGRAM></PROGRAM> tags gave the following output:\n
                                {output}

                                Now determine if the final_answer was correct based on the ground_truth_answer and the program output.
                                ""","role": "user"}],
                    temperature=0.0
                )

            except Exception as e:

                sys.stdout = old_stdout

                response = completion(
                    api_key=apikey,
                    base_url="https://drchat.xyz",
                    model = self.language_model,
                    custom_llm_provider="openai",
                    messages = [{ "content": f"""
                                YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
                                Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct. 
                                ""","role": "user"},
                                { "content": f"""
                                Problem Subject: "{question_subject}", Problem Content: "{question_content}"

                                First, given the ground truth answer, just state the actual answer that is boxed. This is the actual answer that is extracted from the ground truth answer\n
                                Next, compare the final_answer and the ground_truth answer. Don't be strict on the format but check the content. Make sure to consider the context of the question when making your decision . 
                                Is the final_answer correct, given the ground truth answer (ground_truth_answer)? 
                                Think step by step and write a short explanation for whether the two answers match and if further clarification is needed.\n
                                final_answer: {final_answer}\n
                                ground_truth_answer: {ground_truth_answer}\n
                                ""","role": "user"}],
                    temperature=0.0
                )
        
        response = completion(
            api_key=apikey,
            base_url="https://drchat.xyz",
            model = self.language_model,
            custom_llm_provider="openai",
            messages = [{ "content": f"""
                        YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
                        Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct. 
                        ""","role": "user"},
                        { "content": f"""
                        Problem Subject: "{question_subject}", Problem Content: "{question_content}"

                        Here is the final answer and the ground truth answer:\n
                        "final_answer": "{final_answer}", \n"ground_truth_answer": "{ground_truth_answer}"

                        You have determined whether the final_answer is correct or not:\n
                        {response.choices[0].message.content}

                        Given all of this context, Your generated output should be one of the following depending on the correctness of the final_answer:\n
                        {self.valid_correctness}\n\n
                        
                        Your response to this prompt should only be ONE word. It should be one of the three options above. Do not include any other text in your generated response.\n
                        Your generated output will be processed externally.
                        ""","role": "user"}],
            temperature=0.0
        )
        
        return response.choices[0].message.content
    



