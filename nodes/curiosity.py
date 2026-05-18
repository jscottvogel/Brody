import json
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm, CURIOSITY_QUEUE_MAX
from state import ConsciousnessState

class CuriosityOutput(BaseModel):
    selected_question: str = Field(description="The single open question prioritized for immediate exploration")
    exploration_targets: List[str] = Field(description="2-3 concrete exploration targets (sensor inputs to seek or self-reflections to ponder)")
    narrative_note: str = Field(description="A brief note reflecting on the current focus of curiosity to append to the narrative")

def curiosity_node(state: ConsciousnessState) -> dict:
    open_questions = state.get("open_questions", [])
    world_confidence = state.get("world_confidence", {})
    self_confidence = state.get("self_confidence", {})
    contradictions = state.get("contradictions", [])
    curiosity_queue = state.get("curiosity_queue", [])
    narrative = state.get("narrative", "")
    
    # Filter for unresolved contradictions
    unresolved_contradictions = [c for c in contradictions if not c.get("resolved", True)]
    
    if not open_questions and not unresolved_contradictions:
        return {} # Nothing specific to be curious about right now
        
    system_prompt = (
        "You are the curiosity driver of an AI robot. Your task is to review the current "
        "open questions, confidence levels (world and self), and unresolved contradictions.\n"
        "1. Prioritize ONE question or contradiction to explore next. Favor exploring areas "
        "with low confidence, unresolved contradictions, or questions that branch into interesting sub-questions.\n"
        "2. Generate 2 to 3 concrete exploration targets based on your selection. These targets "
        "can be specific sensor inputs to seek out or specific avenues for deep self-reflection.\n"
        "3. Write a brief note about what you are currently curious about, which will be appended to your internal narrative."
    )
    
    human_prompt = (
        f"Open Questions:\n{json.dumps(open_questions, indent=2)}\n\n"
        f"World Confidence:\n{json.dumps(world_confidence, indent=2)}\n\n"
        f"Self Confidence:\n{json.dumps(self_confidence, indent=2)}\n\n"
        f"Unresolved Contradictions:\n{json.dumps(unresolved_contradictions, indent=2)}\n\n"
        "Based on these factors, select the most pressing question or contradiction, generate exploration targets, "
        "and write a narrative note."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = reasoning_llm.with_structured_output(CuriosityOutput)
    response = structured_llm.invoke(messages)
    
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
