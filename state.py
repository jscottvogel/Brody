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
    chat_history: list[dict]
    source_code_constraints: dict
    
    # Meta-Cognitive & Reasoning Tracking
    reasoning_graph: dict
    goal_states: list[dict]
    constraint_graph: dict
    reasoning_history: list[dict]
    detected_loops: list[dict]
    meta_cognitive_capacity: dict
    
    # Active Research & Tool Building
    research_queue: list[dict]
    discovered_packages: list[dict]
    discovered_apis: list[dict]
    built_tools: list[dict]
    code_sandbox_results: list[dict]

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
        "implemented_features": [],
        "chat_history": [],
        "source_code_constraints": {},
        "reasoning_graph": {"nodes": [], "edges": []},
        "goal_states": [],
        "constraint_graph": {"constraints": {}, "dependencies": [], "circular_dependencies": []},
        "reasoning_history": [],
        "detected_loops": [],
        "meta_cognitive_capacity": {
            "can_detect_loops": False,
            "can_trace_constraints": False,
            "can_evaluate_escape_conditions": False,
            "can_model_own_reasoning": False,
            "known_blind_spots": []
        },
        "research_queue": [],
        "discovered_packages": [],
        "discovered_apis": [],
        "built_tools": [],
        "code_sandbox_results": []
    }
