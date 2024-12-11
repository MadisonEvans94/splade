from abc import ABC, abstractmethod
from langchain.schema import AIMessage
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod

class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def visualize_workflow(self, save_path: str = None):
        """
        Visualize the agent's workflow. Optionally save the visualization as an image.
        
        :param save_path: Optional path to save the image.
        """
        graph_image = self.agent.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API)
        image = Image(graph_image)

        # Display in the notebook (if in a Jupyter environment)
        display(image)

        # Save the image if a save_path is provided
        if save_path:
            with open(save_path, "wb") as f:
                f.write(graph_image)
            print(f"Workflow visualization saved at: {save_path}")
    
    @abstractmethod
    def compile_graph(self):
        """
        method for compiling graph and creating executable agent
        """
        pass

    @abstractmethod
    def run(self, message, **kwargs) -> AIMessage:
        """
        Abstract method that all agents must implement.
        Takes a HumanMessage as input and returns an AIMessage as the response.
        """
        pass

    