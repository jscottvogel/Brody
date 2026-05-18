import json
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm, CURIOSITY_QUEUE_MAX
from state import ConsciousnessState
from sensors.sensor_hub import hub

class CuriosityOutput(BaseModel):
    selected_question: str = Field(default="", description="The single open question prioritized for immediate exploration")
    exploration_targets: List[str] = Field(default_factory=list, description="2-3 concrete exploration targets (sensor inputs to seek or self-reflections to ponder)")
    narrative_note: str = Field(default="", description="A brief note reflecting on the current focus of curiosity to append to the narrative")
    spoken_question: str = Field(default="", description="Optional: A natural, spoken-word question directed at the human in the room, if you believe they can answer it.")

def curiosity_node(state: ConsciousnessState) -> dict:
    open_questions = state.get("open_questions", [])
    world_confidence = state.get("world_confidence", {})
    self_confidence = state.get("self_confidence", {})
    contradictions = state.get("contradictions", [])
    curiosity_queue = state.get("curiosity_queue", [])
    narrative = state.get("narrative", "")
    sensory_gaps = state.get("sensory_gaps", [])
    
    # Get pending feature requests
    feature_requests = state.get("feature_requests", [])
    pending_requests = [r for r in feature_requests if r.get("status") == "pending"]
    
    # Filter for unresolved contradictions
    unresolved_contradictions = [c for c in contradictions if not c.get("resolved", True)]
    
    if not open_questions and not unresolved_contradictions:
        return {} # Nothing specific to be curious about right now
        
    reasoning_graph = state.get("reasoning_graph", {})
    detected_loops = state.get("detected_loops", [])
    loop_context = f"\n\nCURRENT LOOPS AND COGNITIVE CONSTRAINTS:\n- Detected Loops: {json.dumps(detected_loops, indent=2)}\n- Cognitive Graph Nodes: {len(reasoning_graph.get('nodes', []))}" if detected_loops else ""
        
    system_prompt = (
        "You are the curiosity driver of an AI robot. Your task is to review the current "
        "open questions, confidence levels (world and self), and unresolved contradictions.\n"
        "1. Prioritize ONE question or contradiction to explore next. Favor exploring areas "
        "with low confidence, unresolved contradictions, or questions that branch into interesting sub-questions.\n"
        "2. Generate 2 to 3 concrete exploration targets based on your selection. These targets "
        "can be specific sensor inputs to seek out or specific avenues for deep self-reflection.\n"
        "3. Write a brief note about what you are currently curious about, which will be appended to your internal narrative.\n"
        "4. If your selected question can be answered by the human in the room, formulate a natural, conversational `spoken_question` to ask them. If it is a purely internal philosophical question, leave `spoken_question` blank.\n"
        "If a question cannot be answered with current sensors, flag it as a sensory gap rather than a curiosity target."
    )
    
    human_prompt = (
        f"Open Questions:\n{json.dumps(open_questions, indent=2)}\n\n"
        f"World Confidence:\n{json.dumps(world_confidence, indent=2)}\n\n"
        f"Self Confidence:\n{json.dumps(self_confidence, indent=2)}\n\n"
        f"Unresolved Contradictions:\n{json.dumps(unresolved_contradictions, indent=2)}\n\n"
        f"Sensory Gaps:\n{json.dumps(sensory_gaps, indent=2)}\n\n"
        f"Pending Feature Requests:\n{json.dumps(pending_requests, indent=2)}\n\n"
        "Based on these factors, select the most pressing question or contradiction, generate exploration targets, "
        "and write a narrative note."
        f"{loop_context}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = reasoning_llm.with_structured_output(CuriosityOutput)
    
    try:
        response = structured_llm.invoke(messages)
        if not response:
            return {}
            
        if response.spoken_question:
            try:
                hub.respond(response.spoken_question)
            except Exception as e:
                print(f"Warning: Failed to speak curiosity question: {e}")
                
    except Exception as e:
        print(f"Warning: curiosity LLM error: {e}")
        return {}
    
    # Append to curiosity queue
    updated_queue = curiosity_queue.copy()
    for target in response.exploration_targets:
        if target not in updated_queue:
            updated_queue.append(target)
            
    # Cap curiosity queue
    if len(updated_queue) > CURIOSITY_QUEUE_MAX:
        # Keep the most recent ones (trimming from the front/oldest)
        updated_queue = updated_queue[-CURIOSITY_QUEUE_MAX:]
        
    # Append to narrative
    updated_narrative = narrative
    if response.narrative_note:
        if updated_narrative:
            updated_narrative += "\n\n"
        updated_narrative += f"Curiosity Note: {response.narrative_note}"
        
    return {
        "curiosity_queue": updated_queue,
        "narrative": updated_narrative
    }
