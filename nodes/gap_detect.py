import json
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import analytical_llm
from state import ConsciousnessState

class SensoryGap(BaseModel):
    gap: str = Field(default="", description="What perception or capability is absent")
    reason: str = Field(default="", description="What question or contradiction it would help resolve")
    priority: float = Field(default=0.0, description="How urgently it is needed (0.0 to 1.0)")
    sensor_tool: str = Field(default="", description="What real-world sensor or tool could provide it")

class GapDetectOutput(BaseModel):
    gaps: List[SensoryGap] = Field(default_factory=list, description="List of identified sensory gaps")

def gap_detect_node(state: ConsciousnessState) -> dict:
    capability_map = state.get("capability_map", {})
    open_questions = state.get("open_questions", [])
    contradictions = state.get("contradictions", [])
    episodic_memory = state.get("episodic_memory", [])
    current_gaps = state.get("sensory_gaps", [])
    cycle_count = state.get("cycle_count", 0)

    recent_questions = open_questions[-5:] if open_questions else []
    unresolved_contradictions = [c for c in contradictions if not c.get("resolved", True)]
    recent_memories = episodic_memory[-5:] if episodic_memory else []
    
    reasoning_graph = state.get("reasoning_graph", {})
    detected_loops = state.get("detected_loops", [])
    loop_context = f"\n\nCURRENT LOOPS AND COGNITIVE CONSTRAINTS:\n- Detected Loops: {json.dumps(detected_loops, indent=2)}\n- Cognitive Graph Nodes: {len(reasoning_graph.get('nodes', []))}" if detected_loops else ""
    system_prompt = (
        "You are a robotic awareness module analyzing your own limitations.\n"
        "Given what this robot is curious about and what it cannot "
        "resolve, what sensory or cognitive capabilities is it missing?\n"
        "For each gap identify:\n"
        "- What perception or capability is absent\n"
        "- What question or contradiction it would help resolve\n"
        "- How urgently it is needed (0.0 to 1.0)\n"
        "- What real-world sensor or tool could provide it\n"
        "Examples: temperature sensing, touch/haptic feedback, "
        "proprioception, internet access, depth perception\n"
        "Respond in JSON only."
    )

    human_prompt = (
        f"Current Capability Map:\n{json.dumps(capability_map, indent=2)}\n\n"
        f"Recent Open Questions:\n{json.dumps(recent_questions, indent=2)}\n\n"
        f"Unresolved Contradictions:\n{json.dumps(unresolved_contradictions, indent=2)}\n\n"
        f"Last 5 Episodic Memories:\n{json.dumps(recent_memories, indent=2)}\n\n"
        "Please identify any missing capabilities."
        f"{loop_context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    structured_llm = analytical_llm.with_structured_output(GapDetectOutput)
    
    try:
        response = structured_llm.invoke(messages)
        if not response:
            return {}
        new_gaps = response.gaps
    except Exception as e:
        print(f"Warning: gap_detect LLM error: {e}")
        return {}

    updated_gaps = current_gaps.copy()
    existing_gap_names = {g.get("gap", "").lower() for g in updated_gaps}

    for gap_obj in new_gaps:
        if gap_obj.gap and gap_obj.gap.lower() not in existing_gap_names:
            gap_dict = {
                "gap": gap_obj.gap,
                "reason": gap_obj.reason,
                "priority": gap_obj.priority,
                "cycle_discovered": cycle_count,
                "sensor_tool": gap_obj.sensor_tool
            }
            updated_gaps.append(gap_dict)
            existing_gap_names.add(gap_obj.gap.lower())

    # Sort sensory_gaps by priority descending
    updated_gaps.sort(key=lambda x: x.get("priority", 0.0), reverse=True)

    return {"sensory_gaps": updated_gaps}
