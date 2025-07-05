import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import requests
import json

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class FinancialSituationMemory:
    def __init__(self, name, config):
        # Check if embedding is supported for this backend
        self.embedding_supported = self._is_embedding_supported(config["backend_url"])

        if not self.embedding_supported:
            if "deepseek.com" in config["backend_url"]:
                if not SENTENCE_TRANSFORMERS_AVAILABLE and not os.getenv("HF_TOKEN"):
                    print(
                        "Warning: DeepSeek doesn't support embeddings. To enable memory functionality:"
                    )
                    print("  Option 1: pip install sentence-transformers (recommended)")
                    print("  Option 2: Add HF_TOKEN to .env file for HuggingFace API")
                else:
                    print("Warning: Embedding setup failed despite available options")
            else:
                print(
                    f"Warning: Embedding not supported for {config['backend_url']}. Memory functionality will be disabled."
                )
            self.client = None
            self.chroma_client = None
            self.situation_collection = None
            return

        # Determine embedding method
        if config["backend_url"] == "http://localhost:11434/v1":
            self.embedding_method = "ollama"
            self.embedding = "nomic-embed-text"
            api_key = None  # Ollama doesn't need API key
        elif "deepseek.com" in config["backend_url"]:
            # Prefer local embeddings over API
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_method = "local"
                self.embedding = "all-MiniLM-L6-v2"  # Small, fast model
                self.embedding_model = None  # Will be loaded lazily
            elif os.getenv("HF_TOKEN"):
                self.embedding_method = "huggingface"
                self.embedding = (
                    "sentence-transformers/all-MiniLM-L6-v2"  # Free HF model
                )
            api_key = self._get_api_key_for_backend(config["backend_url"])
        else:
            self.embedding_method = "openai"
            self.embedding = "text-embedding-3-small"
            # Get API key based on provider
            api_key = self._get_api_key_for_backend(config["backend_url"])

        if self.embedding_method not in ["huggingface", "local"]:
            self.client = OpenAI(base_url=config["backend_url"], api_key=api_key)
        else:
            self.client = (
                None  # HuggingFace/Local uses direct API calls or local models
            )

        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def _is_embedding_supported(self, backend_url: str) -> bool:
        """Check if the backend supports embedding models."""
        # DeepSeek doesn't support embedding models, but we can use alternatives
        if "deepseek.com" in backend_url:
            # Check for local sentence-transformers first (most reliable)
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                return True
            # Fallback to HuggingFace token
            return os.getenv("HF_TOKEN") is not None
        # Add other backends that don't support embeddings here
        return True

    def _get_api_key_for_backend(self, backend_url: str) -> str:
        """Get the appropriate API key based on the backend URL."""
        if "deepseek.com" in backend_url:
            return os.getenv("DEEPSEEK_API_KEY")
        elif "openai.com" in backend_url:
            return os.getenv("OPENAI_API_KEY")
        elif "openrouter.ai" in backend_url:
            return os.getenv("OPENROUTER_API_KEY")
        elif "localhost" in backend_url:
            return None  # Ollama doesn't need API key
        else:
            # Default to OpenAI API key for unknown backends
            return os.getenv("OPENAI_API_KEY")

    def get_embedding(self, text):
        """Get embedding for a text"""
        if not self.embedding_supported:
            return None

        if self.embedding_method == "huggingface":
            return self._get_huggingface_embedding(text)
        else:
            response = self.client.embeddings.create(model=self.embedding, input=text)
            return response.data[0].embedding

    def _get_huggingface_embedding(self, text):
        """Get embedding from HuggingFace Inference Providers API using the client library"""
        try:
            # Try to use the huggingface_hub client library
            from huggingface_hub import InferenceClient

            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variables")

            # Create client with HF Inference provider
            client = InferenceClient(
                provider="hf-inference",
                api_key=hf_token,
            )

            # Use feature_extraction method
            result = client.feature_extraction(
                text,
                model=self.embedding,
            )

            # The result should be a list of floats (the embedding vector)
            # Handle numpy arrays and lists
            import numpy as np

            if hasattr(result, "tolist"):
                # It's a numpy array, convert to list
                return result.tolist()
            elif isinstance(result, list) and len(result) > 0:
                # If it's a 2D array, take the first row
                if isinstance(result[0], list):
                    return result[0]
                else:
                    return result
            else:
                raise Exception(f"Unexpected HuggingFace response format: {result}")

        except ImportError:
            # Fallback to direct HTTP requests if huggingface_hub is not available
            return self._get_huggingface_embedding_http(text)

    def _get_huggingface_embedding_http(self, text):
        """Fallback method using direct HTTP requests"""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")

        # Use the HF Inference API directly (not the router)
        api_url = f"https://api-inference.huggingface.co/models/{self.embedding}"
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        }

        # Send request with proper payload
        payload = {"inputs": text}
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code != 200:
            # Try to get more detailed error info
            try:
                error_detail = response.json()
                error_msg = error_detail.get("error", response.text)
            except:
                error_msg = response.text
            raise Exception(
                f"HuggingFace API error: {response.status_code} - {error_msg}"
            )

        # Parse the response
        result = response.json()

        # Handle response format
        if isinstance(result, list) and len(result) > 0:
            # Direct list format
            if isinstance(result[0], list):
                return result[0]  # Take first embedding if batch
            else:
                return result
        else:
            raise Exception(f"Unexpected HuggingFace response format: {result}")

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""
        if not self.embedding_supported:
            print(
                "Warning: Cannot add situations - embedding not supported for this backend."
            )
            return

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using embeddings"""
        if not self.embedding_supported:
            # Return empty list when embedding is not supported
            return []

        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
