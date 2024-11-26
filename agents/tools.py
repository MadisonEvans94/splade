# agents/tools.py

from langchain.agents import Tool


def calculator_tool(input: str) -> str:
    try:
        # Evaluate the mathematical expression
        result = str(eval(input))
        return result
    except Exception as e:
        return f"Error calculating {input}: {e}"


def search_tool(input: str) -> str:
    # Simulate a search action (since we don't have actual search capability)
    return f"Search results for '{input}': [This is a simulated search result.]"


# Define the tools
calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Useful for performing mathematical calculations."
)

search = Tool(
    name="Search",
    func=search_tool,
    description="Useful for searching information on the internet."
)

# List of tools
TOOLS = [calculator, search]
