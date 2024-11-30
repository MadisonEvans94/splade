from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from agents.base_agent import Agent


class RAGAgent(Agent):
    """
    A wrapper for the LangChain RAG-based agent executor.
    """

    def __init__(self, llm, tools, memory: ConversationBufferMemory):
        """
        Initialize the RAG agent.

        :param llm: The language model.
        :param tools: List of tools used by the agent.
        :param memory: Memory object to maintain chat history.
        """
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
        )

    def run(self, message: HumanMessage) -> AIMessage:
        """
        Process a HumanMessage and return an AIMessage response.
        """
        # Invoke the executor with the user's message
        response = self.agent_executor.invoke(input=message.content)

        # Handle the response and wrap it in an AIMessage
        if isinstance(response, dict) and "output" in response:
            return AIMessage(content=response["output"])
        return AIMessage(content=str(response))
