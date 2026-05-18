import uuid
import datetime
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_anthropic import AnthropicEmbeddings

class LangChainEmbeddingAdapter(EmbeddingFunction):
    """Adapter to use LangChain embeddings with native ChromaDB client."""
    def __init__(self, lc_embedder):
        self.lc_embedder = lc_embedder
        
    def __call__(self, input: Documents) -> Embeddings:
        return self.lc_embedder.embed_documents(input)

class MemoryStore:
    def __init__(self):
        # Use a local persistent ChromaDB instance
        self.client = chromadb.PersistentClient(path="./.chromadb")
        
        # Initialize Anthropic embeddings via LangChain
        lc_embeddings = AnthropicEmbeddings()
        self.embedder = LangChainEmbeddingAdapter(lc_embeddings)
        
        # Collections
        self.memories = self.client.get_or_create_collection(
            name="memories", 
            embedding_function=self.embedder
        )
        self.self_model_collection = self.client.get_or_create_collection(
            name="self_model", 
            embedding_function=self.embedder
        )

    def store_episodic(self, event: str, emotion_tag: str):
        """Embeds and stores an episodic memory with a timestamp."""
        doc_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        self.memories.add(
            documents=[event],
            metadatas=[{
                "type": "episodic",
                "emotion_tag": emotion_tag,
                "timestamp": timestamp
            }],
            ids=[doc_id]
        )

    def store_semantic(self, topic: str, fact: str):
        """Stores a semantic fact under a topic key. Overwrites if topic exists."""
        # Using topic as the ID to allow upsert to overwrite existing topic facts
        self.memories.upsert(
            documents=[fact],
            metadatas=[{
                "type": "semantic",
                "topic": topic
            }],
            ids=[f"semantic_{topic}"]
        )

    def recall_relevant(self, query: str, n: int = 5) -> list[str]:
        """Returns the top-n most semantically similar memories to the query."""
        results = self.memories.query(query_texts=[query], n_results=n)
        
        if not results or not results['documents'] or not results['documents'][0]:
            return []
            
        return results['documents'][0]

    def recall_self_model(self) -> dict:
        """Returns all stored self-model facts as a dict."""
        results = self.self_model_collection.get()
        
        model = {}
        if results and results["ids"]:
            for i, key in enumerate(results["ids"]):
                model[key] = {
                    "value": results["documents"][i],
                    "confidence": results["metadatas"][i]["confidence"]
                }
        return model

    def update_self_model(self, key: str, value: str, confidence: float):
        """Upserts a self-model fact with confidence score."""
        self.self_model_collection.upsert(
            documents=[value],
            metadatas=[{"confidence": confidence}],
            ids=[key]
        )
