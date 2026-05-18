import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables from a .env file
load_dotenv()

# Define model in one place
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

# reasoning_llm: temperature 0.7 for introspective reasoning
reasoning_llm = ChatAnthropic(
    model=MODEL_NAME,
    temperature=0.7
)

# analytical_llm: temperature 0.3 for structured analysis
analytical_llm = ChatAnthropic(
    model=MODEL_NAME,
    temperature=0.3
)

# Constants
CYCLE_LIMIT = 100
CURIOSITY_QUEUE_MAX = 10

DEFAULT_CAPABILITY_MAP = {
    'vision': {
        'available': True,
        'description': 'Webcam RGB vision, forward facing',
        'limitations': 'No depth, no night vision, single angle'
    },
    'audio_in': {
        'available': True,
        'description': 'Microphone, transcription via Whisper',
        'limitations': 'No speaker ID, no sound localization'
    },
    'audio_out': {
        'available': True,
        'description': 'Text to speech via pyttsx3',
        'limitations': 'Synthetic voice, no emotional tone'
    },
    'touch': {
        'available': False,
        'description': 'No tactile sensors',
        'limitations': 'Cannot feel pressure, texture, or contact'
    },
    'temperature': {
        'available': False,
        'description': 'No thermal sensors',
        'limitations': 'Cannot detect ambient or surface heat'
    },
    'movement': {
        'available': False,
        'description': 'Stationary platform',
        'limitations': 'Cannot navigate, turn, or interact physically'
    },
    'internet': {
        'available': False,
        'description': 'No real-time web access tool',
        'limitations': 'Cannot search the web or fetch external data beyond LLM training'
    },
    'proprioception': {
        'available': False,
        'description': 'No internal body awareness',
        'limitations': 'Cannot sense battery, processing load, or hardware state'
    }
}

__all__ = ["reasoning_llm", "analytical_llm", "CYCLE_LIMIT", "CURIOSITY_QUEUE_MAX", "DEFAULT_CAPABILITY_MAP", "CAMERA_INDEX"]
