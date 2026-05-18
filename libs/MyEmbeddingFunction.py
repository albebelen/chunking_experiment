from typing import Dict, Any
import ollama 
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

@register_embedding_function
class MyEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434"):
        """
        Initializes the Ollama embedding client.
        :param model: The Ollama model name (e.g., 'qwen3-embedding:0.6b', 'nomic-embed-text')
        :param host: Optional url if Ollama is running on a different port/machine.
        """
        self.model = model
        self.host = host
        # Create a dedicated Ollama client instance
        self.client = ollama.Client(host=self.host)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Chroma automatically calls this method during .add() and .query().
        It passes a list of strings (`Documents`) and expects a list of float lists (`Embeddings`).
        """
        if not input:
            return []

        # Request batch embeddings from the Ollama server
        response = self.client.embed(
            model=self.model,
            input=input
        )
        
        # Ollama returns a dict containing an 'embeddings' key with the raw lists
        return response["embeddings"]

    @staticmethod
    def name() -> str:
        return "my-ollama-ef"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model, host=self.host)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return MyEmbeddingFunction(
            model=config['model'], 
            host=config.get('host', 'http://127.0.0.1:11434')
        )