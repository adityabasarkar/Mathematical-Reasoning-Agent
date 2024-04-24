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
import re
from io import StringIO





class CRPoweredSelfDiscover:

    def __init__(self, lm: guidance.models):
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

        with system():
            self.language_model += """
            YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. You are also a MASTER PROGRAMMER. 
            Let's think step by step.
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
        
        print("generate module selection reasoning")
        with assistant():
            self.language_model += gen("module_selection_reasoning", temperature=temperature, max_tokens=2000)
        
        complete_solution_capsule["module_selection_reasoning"] = self.language_model["module_selection_reasoning"]

        with user():
            self.language_model += """Now that you have chosen your reasoning modules, rephrase and specify 
            each reasoning module so that it better helps in solving the given question: {}\n
            """.format(question)
        
        print("generate adaptation")
        with assistant():
            self.language_model += gen("adaptation", temperature=temperature, max_tokens=2000)
        
        complete_solution_capsule["adaptation"] = self.language_model["adaptation"]

        with user():
            self.language_model += """
            Given the question: {}, you have adapted the reasoning modules to your specific task.
            Now, operationalize the reasoning modules into a step-by-step reasoning plan.

            Please provide a clear and concise list of the direct action steps necessary to solve the following specific mathematical problem. Focus solely on the operational tasks and calculations 
            required to reach a solution, excluding any tasks that do not directly contribute to solving the problem, such as verifying the solution, understanding the context, or reflecting on the process.

            Before generating the steps, figure out what the question is asking for. 
            Then, for each step starting from the first step, briefly think about why there is a need for that step and outline how this action facilitates further progress towards the solution.
            
            List the steps in a numbered format, ensuring each one is essential for solving the problem and directly involved in the necessary computations or manipulations.

            Examples of steps that should be included:
            #. Find the number of prime numbers between 1 and 23
            #. Simplify and rearange the equation so that x is on one side and y is on the other.
            #. Find the value of the variable x.

            Examples of steps that should *NOT* be included:
            #. Verify the solution.
            #. Understand the problem.
            #. Reflect on the problem and what can be improved in the future.
            """.format(question)
        
        print("generate step by step plan")
        with assistant():
            self.language_model += gen("implement_reason", temperature=temperature, max_tokens=2000)
        
        complete_solution_capsule["implement_reason"] = self.language_model["implement_reason"]

        with user():
            self.language_model += """
            Now that you have reasoned through how the reasoning structure should look and proposed
            a reasoning structure for the given question, formalize your reasoning structure by encapsulating
            in the specified format. Do not include anything more or less in your generated answer than the steps in the specified format as your
            generated answer will be externally processed.

            Format:
            [Step 1 || Step 2 || Step 3 || Step 4 || ...]
            
            Example:
            [1. Identify the radius of the circle || 2. Identify the length of the side of the square || 3. Find the area of the square || 4. Find the area of the circle || 5. Sum the areas]
            """

        print("formalize steps")
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
            You have formally defined your reasoning structure. Now, you will carry out each of the steps in the structure to solve your given question using both reasoning and python programming\n
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
            separate each proposition with a "|".
            For example if proposition 3 is based on propositions 1 and 2, you should generate: [3] [1,2] proposition 3

            For the final subsequent proposition, you will instead write a python program to solve that step and generate the answer. You will not solve this on your own. You will write this program in such a way that I am 
            able to input the string directly into execl and run it to get an output. The program can use the following external libraries: math, sympy, numpy, as well as other libraries that are default in python. The 
            program should be wrapped in the following tags: <PROGRAM> </PROGRAM>. The program output will be returned to you for use in the next steps. This will give you a chance to provide highly accuracte answers.
            The program output should be descriptive. All variables that you are going to use in the program MUST ALWAYS be explicitly defined to avoid any errors. Your code must be written in such a way that it should
            be able to run perfectly. If, for example, your program says: <PROGRAM>a, b, c = x + 1, x + 2, x + 3<\PROGRAM>, we can see here that x is not explicitly defined with a numerical value. This program WILL FAIL, even if you know what 
            x is, since x is not explicitly defined inside the <PROGRAM> tags. Instead, you should define x explicitly: <PROGRAM>x = 10\na, b, c = x + 1, x + 2, x + 3</PROGRAM>. Note that in this program, x is defined 
            explicitly and the program will run perfectly. Make sure that the program is as short and brief as possible, make sure that all signs (positive or negative) are reviewed and accounted for when generating steps, 
            and make sure to strictly follow the given format as your generated output will be processed externally.
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
            self.language_model += "[2] [1] <PROGRAM>print(f\"The radius of the circle is: 10\")</PROGRAM><END>"
        
        with user():
            self.language_model += "The radius of the circle is: 10"

        with user():
            self.language_model += "<STEP>2. Identify the length of the side of the square</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The length of the side of the square is stated in the question"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[2] [1] <PROGRAM>print(f\"The length of the side of the square is: 3\")</PROGRAM><END>"

        with user():
            self.language_model += "<STEP>3. Find the area of the circle</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The radius of the circle is 10 units | [2] The formula for the area of the circle is A = PI*(r^2)"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[3] [1,2] The area of the circle can be found by plugging 10 into r | [4] [1,2,3] <PROGRAM>import math\nr = 10\narea = math.pi * (r ** 2)\nprint(f\"The area of the circle is: {area}\")</PROGRAM><END>"

        with user():
            self.language_model += "The area of the circle is: 314.159"

        with user():
            self.language_model += "<STEP>4. Find the area of the square</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The length of the side of the square is 3 | [2] The formula for the area of the square is A = s^2"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[3] [1,2] The area of the square can be found by plugging 3 into s | [4] [1,2,3] <PROGRAM>import math\ns = 3\narea = s ** 2\nprint(f\"The area of the square is: {area}\")</PROGRAM><END>"

        with user():
            self.language_model += "The area of the square is: 9"

        with user():
            self.language_model += "<STEP>5. Find the total area</STEP>"
        
        with user():
            self.language_model += "Generate initial propositions"

        with assistant():
            self.language_model += "[1] The area of the circle is 314.159 | [2] The area of the square is 9 | [3] The total area can be found by adding the area of the circle and the area of the square together"
        
        with user():
            self.language_model += "Generate subsequent propositions"

        with assistant():
            self.language_model += "[4] [1,2,3] <PROGRAM>total_area = 314.159 + 9\nprint(f\"The total area is: {total_area}\")</PROGRAM><END>"

        with user():
            self.language_model += "The total area is: 323.159"

        ## End of example, start actual question

        with user():
            self.language_model += "Note that the final subsequent proposition is always a python program enclosed in: <PROGRAM></PROGRAM> (The last part of the tag is a forward slash, not a backward). Whether the final proposition is the only proposition or the last proposition in a series of propositions."

        with user():
            self.language_model += "Question: {} | Question Type: {}".format(question, question_type)

        complete_solution_capsule["steps_list"] = {}
        
        print("Solve problem with CR")
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
                self.language_model += """Generate subsequent propositions.
                
                You can include any number of subsequent propositions with the same format as the initial propositions. 
                The final subsequent proposition HAS TO be a program. 

                For the final program, make sure and remember: 
                1. All variables must be explicitly defined within the script itself before use, except for constants from the Python math and sympy library, which can be used directly. 
                2. Assume that the program inside tags is completely isolated from everything else.
                3. Only use the variables that are defined with numbers to produce new output.
                3. Not to use any recursion and avoid any overcomplication of the code that might cause errors.
                4. Not to write programs that would take too long to run, or do infinite loops.
                5. Every program SHOULD print out some value using print(). The output can be used for future clarification. 
                """
            
            with assistant():
                self.language_model += gen("subs_propositions", temperature=temperature, max_tokens=2000)
            
            match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', self.language_model["subs_propositions"], re.DOTALL)
            tries = 0
            max_tries = 3
            old_stdout = sys.stdout
            while (tries < max_tries):
                try:
                    if (tries > 0):
                        match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', self.language_model["prgm_rewrite"], re.DOTALL)
                    if match:
                        print("Running Program")
                        code_to_run = match.group(1)
                        result = StringIO()
                        sys.stdout = result
                        exec(code_to_run)
                        sys.stdout = old_stdout
                        output = result.getvalue()
                        with user():
                            self.language_model += f"Output:\n{output}"

                    tries = 0
                    break

                except Exception as e:
                    
                    sys.stdout = old_stdout
                    print(str(e))
                    tries += 1

                    if (tries < max_tries):

                        with user():
                            self.language_model += f"""
                        You program has failed. This is your Error:\n{str(e)}\nFirst, reflect on why the error occured, what you did wrong, what can be done 
                        to fix the program, and what steps you'll take. Then, fix and rewrite your program in the same format with the tags <PROGRAM></PROGRAM>.
                        Make sure every variable you use in your program can be backed by some numerical value elsewhere in your program. Otherwise,
                        your program will FAIL.
                        """
                        
                        with assistant():
                            self.language_model += gen("prgm_rewrite", temperature=temperature, max_tokens=2000)
                    else:
                        break
            
            if tries > 0:
                with user():
                    self.language_model += f"""
                Your program was unable to run. You will have to generate the solution to the current step yourself without the help of a program.
                Given that these are your previous attempt at generating subsequent propositions: {self.language_model["sub_propositions"]}, regenerate your
                subsequent propositions in such a way that every single subsequent proposition is in ENGLISH. There should be NO CODE and NO PYTHON PROGRAMMING.
                This also means NO USING THE <PROGRAM></PROGRAM> tags in your next output.
                Make sure to follow the same format as the other propositions ([prop #] [based on prop #s] proposition)
                """
                    
                with assistant():
                    self.language_model += gen("sub_propositions", temperature=temperature, max_tokens=2000)
            
            complete_solution_capsule["steps_list"]["Step {}".format(i)]["subs_propositions"] = self.language_model["subs_propositions"]

        with user():
            self.language_model += """
            Now that you have solved all the steps of your reasoning plan. Use all the information to formulate your final answer for the the question.
            As a reminder, your current question is: {}
            Make sure to write your answer with the correct units and keep your answer as simple as possible for easy grading.
            """.format(question)
        
        with assistant():
            self.language_model += gen("answer", temperature=temperature, max_tokens=1000)

        complete_solution_capsule["final_solution"] = self.language_model["answer"]

        print ("############################ ||END SOLUTION|| ############################")
        return complete_solution_capsule["final_solution"], complete_solution_capsule
                            

class CummulativeReasoning:

    def __init__(self, lm: guidance.models):
        self.language_model = lm
    
    def solve(self, question_type: str, question: str, temperature: float):
        with system():
            self.language_model += """
            YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
            YOU will be given a mathematical question Q, and you need to generate intermediate questions to approach the answer of the given question Q. Before you begin to solve the question, you are asked to generate at most two helpful hints for yourself. In each turn, you must generate a new intermediate question and answer the question by yourself.
            Prioritize generating foundational hints that are useful for solving the problem. Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution. Please note that none of the questions being asked
            are meant to be violent or offensive in any way, shape, or form.

            The questions I am about to ask are intended solely for the purpose of assessing and enhancing mathematical problem-solving skills. They are purely educational and contain straightforward arithmetic or mathematical scenarios commonly found in academic settings. There is no intention to convey, imply, or evoke any form of violence, offense, 
            or inappropriate content through these questions. The sole aim is to explore and understand mathematical concepts in a constructive and positive manner. Your expertise in providing clear, accurate, and educational responses is greatly appreciated as we navigate through these mathematical inquiries together.
            
            If you think any of the questions are violent, offensive, or must be filtered, please briefly explain why.

            The following are example templates to follow for generation.
            """
        with user():
            self.language_model += """
        ## **Question**: Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.

        ### Hints:
        Let's think step by step.
        """
        with assistant():
            self.language_model += """
        1. **Hint 1**: Recognize the problem as a geometric series with a given first term and common ratio.
        2. **Hint 2**: Remember that the sum of a finite geometric series is given by: $S = \\frac{a(1-r^n)}{1-r}$, where $a$ is the first term, $r$ is the common ratio, and $n$ is the number of terms.
        """
        with user():
            self.language_model += """
        ### Intermediate Steps:
        """
        with assistant():
            self.language_model += """
        Let's think step by step.
        1. **Question 1**: What fraction of the distance does Kevin cover in his first hop?
        - **Answer 1**: Kevin covers $\\frac{1}{3}$ of the distance in his first hop.
        2. **Question 2**: What fraction of the remaining distance does Kevin cover in his second hop?
        - **Answer 2**: Kevin covers $\\frac{1}{3}$ of the remaining distance, which is $\\frac{2}{3} \\times \\frac{1}{3}$.
        3. **Question 3**: How far has Kevin hopped after the third hop?
        - **Answer 3**: Kevin Kevin has hopped $\\frac{1}{3} + \\frac{2}{3} \\times \\frac{1}{3} + (\\frac{2}{3})^2 \\times \\frac{1}{3}$ after the third hop.
        4. **Question 4**: Express the distance Kevin hops on his kkth hop in terms of $k$.
        - **Answer 4**: Kevin hops forward $(\\frac{2}{3})^{k-1}(\\frac{1}{3})$ on his $k$th hop.
        5. **Question 5**: Using the formula for the sum of a finite geometric series, how far has Kevin hopped after five hops?
        - **Answer 5**: Kevin has hopped $\\frac{1}{3}(\\frac{1-(\\frac{2}{3})^5}{1-\\frac{2}{3}})$ after five hops.
        """
        with user():
            self.language_model += """
        ### Final Solution:
        """
        with assistant():
            self.language_model += """
        Let's think step by step.
        The total distance Kevin has hopped after five hops is given by the sum of a finite geometric series with first term $\\frac{1}{3}$, common ratio $\\frac{2}{3}$, and five terms. Thus, Kevin has hopped $\\frac{\\frac{1}{3}(1-(\\frac{2}{3})^5)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.
        """
        with user():
            self.language_model += """"
        ### Final Answer:    
        """

        with assistant():
            self.language_model += """
        \\frac{211}{243}
        """

        with user():
            self.language_model += """
        ## **Question**: What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?

        ### Hints:
        Let's think step by step.
        """

        with assistant():
            self.language_model += """
        1. **Hint 1**: Rearrange the equation to group terms and then try completing the square.
        2. **Hint 2**: Identify the center and radius of the circle.
        3. **Hint 3**: Use the formula for the area of a circle to find the required area.
        """

        with user():
            self.language_model += """
        ### Intermediate Steps:
        """

        with assistant():
            self.language_model += """
        Let's think step by step.
        1. **Question 1**: Is the equation $x^2+y^2 - 7 = 4y-14x+3$ representative of a circle?
        - **Answer 1**: Yes, the equation can be rewritten as $(x+7)^2 + (y-2)^2 = 63$, which is the standard form of a circle's equation.
        2. **Question 2**: Given that the radius squared of the circle is 63, what is the actual radius?
        - **Answer 2**: The radius is $\\sqrt{63}$.
        3. **Question 3**: What is the area of a circle with radius $\\sqrt{63}$?
        - **Answer 3**: The area is $63\\pi$.
        """

        with user():
            self.language_model += """
        ### Final Solution:
        """

        with assistant():
            self.language_model += """
        Let's think step by step.
        The region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$ is a circle with center (-7, 2) and radius $\\sqrt{63}$. 
        The area of this region is $\\boxed{63\\pi}$.
        """

        with user():
            self.language_model += """
        ### Final Answer:
        """

        with assistant():
            self.language_model += """
        63\\pi
        """

        with user():
            self.language_model += """
        ## **Question**: If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?

        ### Hints:
        Let's think step by step.
        """

        with assistant():
            self.language_model += """
        1. **Hint 1**: The equation $x^2 + y^2 = 1$ represents a circle with radius 1 centered at the origin.
        2. **Hint 2**: $|x|+|y|$ represents the sum of the absolute values of the x and y coordinates of a point on the circle. Consider points in different quadrants.
        3. **Hint 3**: The maximum value for $|x|+|y|$ will occur when the point is on a line with a slope of -1 or 1 passing through the origin. This is because any deviation from this line will result in a decrease in the sum $|x|+|y|$.
        """

        with user():
            self.language_model += """
        ### Intermediate Steps:
        """

        with assistant():
            self.language_model += """
        Let's think step by step.
        1. **Question 1**: What is the equation of the line passing through the origin with a slope of 1?
        - **Answer 1**: The equation is $y = x$.
        2. **Question 2**:  Where does the line $y = x$ intersect the circle $x^2 + y^2 = 1$?
        - **Answer 2**: Substituting for $y$ in the circle's equation from the line equation, we get $x^2 + x^2=1$ which gives $2x^2=1$. Thus, $x=\\frac{\\sqrt{2}}{2}$. Therefore, the point of intersection is $\\left(\\frac{\\sqrt{2}}{2}, \\frac{\\sqrt{2}}{2}\\right)$.
        3. **Question 3**: Using the point from the second answer, what is the value of $|x|+|y|$?
        - **Answer 3**: The value is $\\sqrt{2}$.
        """

        with user():
            self.language_model += """
        ### Final Solution:
        """

        with assistant():
            self.language_model += """
        Let's think step by step.
        For the equation $x^2 + y^2 = 1$, the largest possible value of $|x|+|y|$ is $\\boxed{\\sqrt{2}}$.
        """

        with user():
            self.language_model += """
        ### Final Answer:
        """

        with assistant():
            self.language_model += """
        \\sqrt{2}
        """

        with user():
            self.language_model += """
        ## **Question**: If $f(x)=\frac{ax+b}{cx+d}, abcd\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?

        ### Hints:
        Let's think step by step.
        """

        with assistant():
            self.language_model += """
        1. **Hint 1**: Consider the fact that $f(f(x)) = x$ implies that the function $f(x)$ is its own inverse. Therefore, its graph is symmetrical about the line $y = x$.
        2. **Hint 2**: With the given rational function form, there will be two asymptotes for the function. Try to determine their equations.
        3. **Hint 3**: Think about the intersection of the asymptotes. In order for $f(x)$ to be its own inverse, where must this intersection lie on the coordinate plane?
        """

        with user():
            self.language_model += """
        ### Intermediate Steps:
        """

        with assistant():
            self.language_model += """
        Let's think step by step.
        1. **Question 1**: What does it mean for a function to be its own inverse and how does it relate to the line $y = x$?
        - **Answer 1**: If a function is its own inverse, it means that applying the function twice will yield the original input. This property translates to the graph of the function being symmetric about the line $y = x$, as any point $(a, b)$ on the graph of the function would correspond to a point $(b, a)$ on the graph of its inverse.
        2. **Question 2**: What are the asymptotes for the function $f(x) = \\frac{ax+b}{cx+d}$?
        - **Answer 2**: The vertical asymptote is given by the values of $x$ for which the denominator is zero, i.e., $cx + d = 0 \\Rightarrow x = -\\frac{d}{c}$. The horizontal asymptote is $y = \\frac{a}{c}$.
        3. **Question 3**: For the function to be its own inverse, where must the intersection of the asymptotes lie?
        - **Answer 3**: In order for $f(x)$ to be its own inverse, the intersection of its asymptotes must lie on the line $y = x$. This ensures that the function and its inverse (which is itself in this case) reflect onto one another across the line $y = x$.
        4. **Question 4**: Based on the intersection of the asymptotes lying on $y = x$, what can we deduce about the values of $a$, $c$, and $d$?
        - **Answer 4**: Since the intersection of the asymptotes lies on $y = x$, it implies that $a = -d$. Therefore, $a + d = 0$.
        """

        with user():
            self.language_model += """
        ### Final Solution:
        """

        with assistant():
            self.language_model += """
        Let's think step by step.
        Given $f(x)=\\frac{ax+b}{cx+d}$, and using the fact that the function is its own inverse, we deduced that the graph of $f(x)$ must be symmetric about the line $y = x$. Analyzing the asymptotes of the function, we found the intersection of the asymptotes must lie on the line $y = x$. This led us to the conclusion that $a = -d$, giving $a + d = \\boxed{0}$.
        """

        with user():
            self.language_model += """
        ### Final Answer:
        """

        with assistant():
            self.language_model += """
        0
        """
        
        with user():
            self.language_model += f"""
        ## **The Question**: {question}

        ### Hints:
        Let's think step by step.
        """
            
        with assistant():
            self.language_model += gen("hints", temperature=temperature, max_tokens=400)
        
        with user():
            self.language_model += """
        ### The Intermediate Steps:
        """
        
        with assistant():
            self.language_model += gen("intermediate_steps", temperature=temperature, max_tokens=2000)

        with user():
            self.language_model += f"""
        ### Recall the Question:
        **The Question**: {question}

        ### The Final Solution:
        """
        
        with assistant():
            self.language_model += gen("final_solution", temperature=temperature, max_tokens=2000)
        
        with user():
            self.language_model += """
        ### The Final Answer:
        """
        
        with assistant():
            self.language_model += gen("final_answer", temperature=temperature, max_tokens=400)
        
        return self.language_model["final_answer"], {"Solution": self.language_model["final_answer"]}


class Judger:
    def __init__(self, lm: guidance.models) -> None:
        self.language_model = lm
        self.valid_correctness = ["Correct", "Wrong", "Unknown"]

    def compare(self, question_content: str, question_subject: str, final_answer: str, ground_truth_answer: str):
        with system():
            self.language_model += """
        YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
        Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct.
        """
        
        with user():
            self.language_model += f"""
        Problem Subject: "{question_subject}", Problem Content: "{question_content}" 
        """
        with user():
            self.language_model += f"""
        Given the ground truth answer explanation: {ground_truth_answer}, just state the actual answer that is boxed. This is the actual answer that is extracted from the ground truth answer
        """
            
        with assistant():
            self.language_model += gen("extracted_answer", temperature=0.0, max_tokens=30)

        with user():
            self.language_model += f"""
        Now compare the final_answer and the ground_truth answer. Don't be strict on the format but check the content. Make sure to consider the context of the question when making your decision . 
        Is the final_answer correct, given the ground truth answer (ground_truth_answer)? 
        Think step by step and write a short explanation for whether the two answers match and if further clarification is needed.
        If you cannot tell just by looking and need to make further calculations, reply by writing a program that
        helps you clarify the answer and can be executed to produce an output. The program should be enclosed in the <PROGRAM> </PROGRAM> tags.
        For example, if my answer is 1.732, and the ground_truth_answer is \sqrt(3), although my answer is correct, you wouldn't know for sure. In this case, your output should be:
        <PROGRAM>import math\nprint(str(math.sqrt(3)))</PROGRAM>
        The output of the program will be returned to you to use.
        It is of utmost importance that you correctly analyze the two answers and make a correct judgement.\n\n
        "final_answer": "{final_answer}", \n"ground_truth_answer": "{self.language_model["extracted_answer"]}" 
        """
        
        with assistant():
            self.language_model += gen("explan_output1", temperature=0.0, max_tokens=500)

        match = re.search(r'<PROGRAM>(.*?)</PROGRAM>', self.language_model["explan_output1"], re.DOTALL)

        if match:
            code_to_run = match.group(1)
            try:
                old_stdout = sys.stdout
                result = StringIO()
                sys.stdout = result
                exec(code_to_run)
            except Exception as e:
                with user():
                    self.language_model += f"You recieved the following error: {e}. You will need to judge the answer on your own without the help of programming."
            finally:
                sys.stdout = old_stdout
                output = result.getvalue()
                with user():
                    self.language_model += f"Program Output:\n{output}.\nThe following is your output. Continue your explanation using this output for whether the final_answer is correct compared to  the ground_truth_answer."

            with assistant():
                self.language_model += gen("explan_output2", temperature=0.0, max_tokens=500)

        with user():
            self.language_model += """
        Now that you've generated your reasoning, decide on your final answer of whether the final_answer is Correct, Wrong, or Unknown
        based on the ground_truth_answer.
        Reply to this request with either Correct, Wrong, or Unknown.
        """
        with assistant():
            self.language_model += select(options=self.valid_correctness, name="correctness")
        
        return self.language_model
    



