import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from agents.conversation_agent import SimpleLLMAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the agent with your OpenAI API key
agent = SimpleLLMAgent(OPENAI_API_KEY=OPENAI_API_KEY)

# Prepare a list of human messages
messages = [
    HumanMessage(content="Hello, how are you?"),
    HumanMessage(content="Can you tell me a joke?"), 
    AIMessage(content="Hello! I'm doing well, thank you for asking. Here's a joke for you: Why did the scarecrow win an award? Because he was outstanding in his field!"),
    HumanMessage(content="That's a good one! Can you tell me another joke?")
]

# Run the agent
response = agent.run(messages)

# Print the AI's response
print(response.content)
