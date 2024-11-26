import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from agents.agent_factory import AgentFactory

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


factory = AgentFactory(OPENAI_API_KEY=OPENAI_API_KEY)

agent_type = input(
    "Enter agent type (conversation_agent, web_search_agent, or graph_agent): ").strip()

try:
    agent = factory.factory(agent_type)
except ValueError as e:
    print(e)
    exit()

# Start interacting with the agent
print("Start interacting with the agent (type 'exit' to stop):")
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user's input to chat history
    human_message = HumanMessage(content=user_input)
    chat_history.append(human_message)

    # Run the agent with the input and chat history
    response = agent.run(chat_history)

    # Save AI's response to chat history
    chat_history.append(response)

    # Print the AI's response
    print(f"AI: {response.content}")
