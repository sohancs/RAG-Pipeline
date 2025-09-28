from langchain.llms.base import LLM
import requests

class OllamaWrapper(LLM):
    """Wrapper to use Ollama models using API calls."""

    model_name: str
    base_url: str

    def _call(self, prompt: str, stop: list[str] = None) -> str :
        api_url = f"{self.base_url}/api/generate"
        payload = {
            "model" : self.model_name,
            "prompt": prompt,
            "temperature" : 0,
            "stream": False
        }
        print (f"Sending request to Ollama model {self.model_name} at {self.base_url}")
        response = requests.post(api_url, json=payload)
        
        return response.json()["response"]

    @property
    def _identifying_params(self) :
        return {"model_name": self.model_name, "base_url": self.base_url}
    
    @property
    def _llm_type(self) -> str :
        return "ollama"