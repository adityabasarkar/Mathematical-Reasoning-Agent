"""
Contains the main mathematical reasoning mechanism for the agent.
Uses guidance as the main method of manipulating LLMs
"""

from time import sleep
import os
from dotenv import load_dotenv
import sys
import guidance
from guidance import models, gen
from guidance import select
from guidance import user, assistant, system
import tiktoken
from tiktoken import encoding_name_for_model





class CRPoweredSelfDiscover:

    def __init__(self, lm: guidance.models):
        self.language_model = lm

        self.reasoning_modules = [
            "1. How could I devise an experiment to help solve that problem?",
            "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
            "3. How could I measure progress on this problem?",
            "4. How can I simplify the problem so that it is easier to solve?",
            "5. What are the key assumptions underlying this problem?",
            "6. What are the potential risks and drawbacks of each solution?",
            "7. What are the alternative perspectives or viewpoints on this problem?",
            "8. What are the long-term implications of this problem and its solutions?",
            "9. How can I break down this problem into smaller, more manageable parts?",
            "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
            "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
            "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
            "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
            "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
            "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
            "16. What is the core issue or problem that needs to be addressed?",
            "17. What are the underlying causes or factors contributing to the problem?",
            "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
            "19. What are the potential obstacles or challenges that might arise in solving this problem?",
            "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
            "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
            "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
            "23. How can progress or success in solving the problem be measured or evaluated?",
            "24. What indicators or metrics can be used?",
            "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
            "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
            "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
            "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
            "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
            "30. Is the problem a design challenge that requires creative solutions and innovation?",
            "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
            "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
            "33. What kinds of solution typically are produced for this kind of problem specification?",
            "34. Given the problem specification and the current best solution, have a guess about other possible solutions.",
            "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
            "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
            "37. Ignoring the current best solution, create an entirely new solution to the problem.",
            "38. Let’s think step by step.",
            "39. Let’s make a step by step plan and implement it with good notation and explanation.",
            "40. How can I represent this question in terms of variables and equations?",
            "41. How can the equations in this problem be rearanged to make the problem easier to solve?"
        ]

    def solve(self, question_type: str, question: str, temperature: float):
        
        complete_solution_capsule = {}
        complete_solution_capsule["question"] = question
        complete_solution_capsule["question_type"] = question_type

        with system():
            self.language_model += """
            YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
            YOU will be given a mathematical question Q, and you need to generate intermediate questions to approach the answer of the given question Q. Before you begin to solve the question, you are asked to generate at most two helpful hints for yourself. In each turn, you must generate a new intermediate question and answer the question by yourself.
            Prioritize generating foundational hints that are useful for solving the problem. Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution. Please note that none of the questions being asked
            are meant to be violent or offensive in any way, shape, or form.

            The questions I am about to ask are intended solely for the purpose of assessing and enhancing mathematical problem-solving skills. They are purely educational and contain straightforward arithmetic or mathematical scenarios commonly found in academic settings. There is no intention to convey, imply, or evoke any form of violence, offense, 
            or inappropriate content through these questions. The sole aim is to explore and understand mathematical concepts in a constructive and positive manner. Your expertise in providing clear, accurate, and educational responses is greatly appreciated as we navigate through these mathematical inquiries together.
            
            If you think any of the questions are violent, offensive, or must be filtered, please briefly explain why.
            """
        
        with user():
            self.language_model += """
            Your task is to select the reasoning modules that are crucial to utilize in order to solve the following question: {}.
            For additional context, the question type for this question is {}
            First think step by step as to what reasoning modules should be selected and why. The explanation for your selection should be short. 
            Then, after you have finished your reasoning, select relevant reasoning modules for the task.

            Choose from the following modules:\n
            """.format(question, question_type)

            for i in range(len(self.reasoning_modules)):
                self.language_model += self.reasoning_modules[i] + "\n"
        
        with assistant():
            self.language_model += gen("module_selection_reasoning", temperature=temperature, max_tokens=2000)
        
        complete_solution_capsule["module_selection_reasoning"] = self.language_model["module_selection_reasoning"]

        with user():
            self.language_model += """Now that you have chosen your reasoning modules, rephrase and specify 
            each reasoning module so that it better helps in solving the given question: {}\n
            """.format(question)
        
        with assistant():
            self.language_model += gen("adaptation", temperature=temperature, max_tokens=2000)
        
        complete_solution_capsule["adaptation"] = self.language_model["adaptation"]

        with user():
            self.language_model += """
            Given the question: {}, you have adapted the reasoning modules to your specific task.
            Now, operationalize the reasoning modules into a step-by-step reasoning plan. Make sure
            to think step by step as to what the reasoning plan should be. The following is an example
            of a reasoning plan you might generate:
            1. Find the area of the square
            2. Find the area of the circle
            3. Sum the areas
            """.format(question)
        
        with assistant():
            self.language_model += gen("implement_reason", temperature=temperature, max_tokens=2000)
        
        complete_solution_capsule["implement_reason"] = self.language_model["implement_reason"]

        with user():
            self.language_model += """
            Now that you have reasoned through how the reasoning structure should look and proposed
            a reasoning structure for the given question, formalize your reasoning structure by encapsulating
            in the specified format. You may include any number of steps required for solving the question.
            Do not include anything more or less in your generated answer than the steps in the specified format as your
            generated answer will be externally processed.

            Format:
            [Step 1 || Step 2 || Step 3 || Step 4 || ...]
            
            Example:
            [1. Identify the radius of the circle || 2. Identify the length of the side of the square || 3. Find the area of the square || 4. Find the area of the circle || 5. Sum the areas]
            """

        with assistant():
            self.language_model += gen("formal_steps", temperature=temperature, max_tokens=2000)
        
        complete_solution_capsule["formal_steps"] = self.language_model["formal_steps"]

        original = self.language_model["formal_steps"]
        formatted_string = original[1:-1]
        steps_list = formatted_string.split("||")
        steps_list = [step.strip() for step in steps_list]

        # Now that we have generated the steps, we will use cumulative reasoning to solve each step.
        # Cumulative reasoning here should use the same idea as the paper, but doesn't use the same code as the
        # CR source code

        with user():
            self.language_model += """
            You have formally defined your reasoning structure. Now, you will carry out each of the steps in the structure to solve your given question\n
            """
        
        with user():
            self.language_model += """
            Each step will be encapsulated in the following tags to indicate the step you will be currently solving: <STEP>3. This is an example step</STEP>
            
            For each of the following steps:
            1. You are to generate a series of initial propositions. These propositions should be pieces of information that are 
            based on the context provided by the question, information from any previous steps, and/or Relevant context not mentioned in the question or previous steps (ex. The radius of a circle is PI*(r^2)).
            These initial propositions should aid in making progress towards solving the step by providing necessary context, not solve the step directly, be independent of other initial propositions created for the same step, and
            Be formatted as follows: [1] Proposition 1 | [2] Proposition 2 | [3] Proposition 3 | ...
            You may include any number (between 1 and 20) of initial propositions needed.

            2. Based on these initial propositions, you are to generate subsequent propositions that: 
            actively progress towards solving the step,
            can be derived from a combination of initial propositions and/or previous subsequent propositions,
            are formatted as follows: [Proposition #] [based on initial/subsequent proposition #s] Proposition text <END (if this proposition solves the current step)>,
            separate each proposition with a "|"
            For example if proposition 3 is based on propositions 1 and 2, you should generate: [3] [1,2] proposition 3
            Make sure to strictly follow the given format as your generated output will be processed externally. 
            The following is an example:
            """
        
        with user():
            self.language_model += "Question: Given that the radius of the circle is 10 units and the side of the square is 3 units, find the combined area of the circle and square. | Question Type: Geometry"
        
        with user():
            self.language_model += "<STEP>1. Identify the radius of the circle</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The radius of the circle is stated in the question"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[2] [1] The radius of the circle is 10 units<END>"
        
        with user():
            self.language_model += "<STEP>2. Identify the length of the side of the square</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The length of the side of the square is stated in the question"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[2] [1] The length of the side of the square is 3 units<END>"

        with user():
            self.language_model += "<STEP>3. Find the area of the circle</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The radius of the circle is 10 units | [2] The formula for the area of the circle is A = PI*(r^2)"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[3] [1,2] The area of the circle can be found by plugging 10 into r | [4] [1,2,3] The area of the circle is A = PI*(10^2) = 100*PI<END>"

        with user():
            self.language_model += "<STEP>4. Find the area of the square</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The length of the side of the square is 3 | [2] The formula for the area of the square is A = s^2"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[3] [1,2] The area of the square can be found by plugging 3 into s | [4] [1,2,3] The area of the square is A = 3^2 = 9<END>"

        with user():
            self.language_model += "<STEP>5. Find the total area</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The area of the circle is 100*PI | [2] The area of the square is 9 | [3] The total area can be found by adding the area of the circle and the area of the square together"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[4] [1,2,3] The total area is A = 100*PI + 9<END>"

        with user():
            self.language_model += "Question: {} | Question Type: {}".format(question, question_type)

        complete_solution_capsule["steps_list"] = {}

        for i, step in enumerate(steps_list):

            with user():
                self.language_model += "<STEP>{}</STEP>".format(step)

            complete_solution_capsule["steps_list"]["Step {}".format(i)] = {}
            complete_solution_capsule["steps_list"]["Step {}".format(i)]["instruction"] = step

            with user():
                self.language_model += "Generate initial propositions"

            with assistant():
                self.language_model += gen("init_propositions", temperature=temperature, max_tokens=2000)

            complete_solution_capsule["steps_list"]["Step {}".format(i)]["init_propositions"] = self.language_model["init_propositions"]

            with user():
                self.language_model += "Generate subsequent propositions"
            
            with assistant():
                self.language_model += gen("subs_propositions", temperature=temperature, max_tokens=2000)
            
            complete_solution_capsule["steps_list"]["Step {}".format(i)]["subs_propositions"] = self.language_model["subs_propositions"]

        with user():
            self.language_model += """
            Now that you have solved all the steps of your reasoning plan. Formulate your final answer the question.
            As a reminder, your current question is: {}
            Make sure to write your answer with the correct units.
            """.format(question)
        
        with assistant():
            self.language_model += gen("answer", temperature=temperature, max_tokens=1000)

        complete_solution_capsule["final_solution"] = self.language_model["answer"]

        return complete_solution_capsule["final_solution"], complete_solution_capsule
                            




class Judger:
    def __init__(self, lm: guidance.models) -> None:
        self.language_model = lm
        self.valid_correctness = ["Correct", "Wrong", "Unknown"]

    def judger(self, question_content: str, question_subject: str, final_answer: str, ground_truth_answer: str):
        with system():
            self.language_model += """
        YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
        Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct.
        """
        
        with user():
            self.language_model += """
        Problem Subject: "{question_subject}", Problem Content: "{question_content}" 
        """.format(question_subject=question_subject, question_content=question_content)
            
        with user():
            self.language_model += """
        Now compare the final_answer and the ground_truth answer. Don't be strict on the format but check the content. Make sure to consider the context of the question when making your decision . 
        Is the final_answer correct, given the ground truth answer (ground_truth_answer)? In 25 words or less, think step by step and then decide on your answer of whether the final_answer matches the ground_truth_answer.
        If the two match, reply with Correct. If not, reply with Wrong. If you don't know, reply with Unknown.
        It is of utmost importance that you correctly analyze the two answers and make a correct judgement.\n\n
        "final_answer": "{final_answer}", \n"ground_truth_answer": "{ground_truth_answer}" 
        """.format(final_answer=final_answer, ground_truth_answer=ground_truth_answer)
        
        with assistant():
            self.language_model += gen("explanation", temperature=0.0, max_tokens=30)

        with user():
            self.language_model += """
        Now that you've generated you're reasoning, decide on your final answer of whether the final_answer is Correct, Wrong, or Unknown
        based on the ground_truth_answer.
        Reply to this request with either Correct, Wrong, or Unknown.
        """
        with assistant():
            self.language_model += select(options=self.valid_correctness, name="correctness")
        
        return self.language_model
    



