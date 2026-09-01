import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def calculate(expression: str) -> str:
    """A simple calculator tool, eval() is not safe."""
    print("tool called")
    try:
        result = str(eval(expression))
        return result
    except Exception as e:
        return f"Error: {e}"

def parse_react_action(agent_response: str):
    """Parse the local ReAct tool call format used by this script."""
    action_match = re.search(r"Action:\s*(\w+)", agent_response)
    action_input_match = re.search(r"Action Input:\s*(.*)", agent_response, re.DOTALL)

    if not action_match or not action_input_match:
        return None, None

    return action_match.group(1).strip(), action_input_match.group(1).strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py '<math question>'")
        return

    user_question = sys.argv[1]

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: Please set your GROQ_API_KEY environment variable.")
        return

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    system_prompt = """
    You are a math assistant.

    IMPORTANT:
    - NEVER output native tool-calling JSON.
    - NEVER output names like tool:calculate.
    - NEVER output a payload like {"name": "tool:calculate", "arguments": {"input": "1+2"}}.
    - NEVER use Groq tool-calling syntax.
    - ONLY use the local plain-text ReAct format below.

    For arithmetic problems, you must use this exact local format:
    Thought: I need to calculate something.
    Action: calculate
    Action Input: 1 + 2

    For final answers, use this exact local format:
    Thought: I have the final answer.
    Final Answer: The answer is 3.

    Rules:
    1. If the user asks for a calculation, prefer the Action format.
    2. The Action line must be exactly: Action: calculate
    3. The Action Input line must be exactly: Action Input: <expression>
    4. Do not add JSON, code fences, or tool metadata.
    5. Only plain text is allowed.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    print(f"User: {user_question}\n")

    max_loops = 5
    loop_count = 0

    while loop_count < max_loops:
        loop_count += 1

        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.3
        )

        agent_response = completion.choices[0].message.content
        print(f"={agent_response}")

        messages.append({"role": "assistant", "content": agent_response})

        if "Final Answer:" in agent_response:
            answer = agent_response.split("Final Answer:")[-1].strip()
            print(f"Final Answer: {answer}")
            break

        tool_name, action_input = parse_react_action(agent_response)
        if tool_name == "calculate":
            observation_result = calculate(action_input)
            print(f"Observation: {observation_result}")
            messages.append({"role": "user", "content": f"Observation: {observation_result}"})
            print(f"Final Answer: {observation_result}")
            break

        print("Observation: Unknown action or malformed response")
        messages.append({"role": "user", "content": "Observation: Unknown action or malformed response"})
        break

    if loop_count == max_loops:
        print("Agent hit max loops and was killed!")


if __name__ == "__main__":
    main()
