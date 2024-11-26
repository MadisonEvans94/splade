import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from agents.agent_factory import AgentFactory

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the AgentFactory with the OpenAI API key
factory = AgentFactory(OPENAI_API_KEY=OPENAI_API_KEY)

# Create a ToolCallingAgent using the factory
agent = factory.factory("conversation_agent")

# Start the conversation loop
print("Start interacting with the agent (type 'exit' to stop):")
messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Add the user's message to the conversation
    messages.append(HumanMessage(content=user_input))

    # Run the agent with the messages
    response = agent.run(messages)

    # Add the AI's response to the conversation
    messages.append(response)

    # Print the AI's response
    print(f"AI: {response.content}")
