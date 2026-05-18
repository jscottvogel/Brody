import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables from a .env file
load_dotenv()

# reasoning_llm: temperature 0.7 for introspective reasoning
reasoning_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.7
)

# analytical_llm: temperature 0.3 for structured analysis
analytical_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.3
)

# Constants
CYCLE_LIMIT = 100
CURIOSITY_QUEUE_MAX = 10

__all__ = ["reasoning_llm", "analytical_llm", "CYCLE_LIMIT", "CURIOSITY_QUEUE_MAX"]
