"""
Contains the main mathematical reasoning mechanism for the agent.
Uses guidance as the main method of 
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

reasoning_modules = [
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
    "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
    "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    "37. Ignoring the current best solution, create an entirely new solution to the problem."
    "38. Let’s think step by step."
    "39. Let’s make a step by step plan and implement it with good notation and explanation."
]

class MathReasoning:

    def __init__(self, lm: guidance.models) -> None:
        pass

    def solve(language_model: models.OpenAIChat, question_type: str, question: str):
        with system():
            language_model += """
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
            language_model += """
            
            """
        