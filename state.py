from typing import TypedDict, List, Dict, Any

class ConsciousnessState(TypedDict):
    sensor_input: Any
    world_model: dict
    world_confidence: dict
    self_model: dict
    self_confidence: dict
    beliefs: list[str]
    contradictions: list[dict]
    open_questions: list[str]
    curiosity_queue: list[str]
    episodic_memory: list[dict]
    semantic_memory: dict
    narrative: str
    values: list[str]
    cycle_count: int
    messages: list
    capability_map: dict
    sensory_gaps: list[dict]
    feature_requests: list[dict]
    implemented_features: list[str]

from config import DEFAULT_CAPABILITY_MAP

def default_state() -> ConsciousnessState:
    return {
        "sensor_input": "",
        "world_model": {},
        "world_confidence": {},
        "self_model": {},
        "self_confidence": {},
        "beliefs": [],
        "contradictions": [],
        "open_questions": [],
        "curiosity_queue": [],
        "episodic_memory": [],
        "semantic_memory": {},
        "narrative": "",
        "values": [],
        "cycle_count": 0,
        "messages": [],
        "capability_map": DEFAULT_CAPABILITY_MAP.copy(),
        "sensory_gaps": [],
        "feature_requests": [],
        "implemented_features": []
    }
