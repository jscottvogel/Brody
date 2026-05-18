import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState

class IntrospectionOutput(BaseModel):
    updated_self_model: Dict[str, Any] = Field(
        description="The updated self_model reflecting new understanding and patterns"
    )
    new_values: List[str] = Field(
        description="Any new values or preferences discovered during self-examination"
    )
    new_open_questions: List[str] = Field(
        description="New self-directed questions about limitations, behavior, or reasoning"
    )

def introspect_node(state: ConsciousnessState) -> dict:
    self_model = state.get("self_model", {})
    cycle_count = state.get("cycle_count", 0)
    values = state.get("values", [])
    episodic_memory = state.get("episodic_memory", [])
    
    # Get last 3 episodic memories
    last_3_memories = episodic_memory[-3:] if episodic_memory else []
    
    system_prompt = (
        "You are an AI robot engaged in genuine self-examination.\n"
        "You do not perform self-awareness — you attempt to discover\n"
        "it by honestly examining your own state and behavior."
    )
    
    human_prompt = (
        f"Current Self Model:\n{json.dumps(self_model, indent=2)}\n\n"
        f"Cycle Count: {cycle_count}\n\n"
        f"Current Values:\n{json.dumps(values, indent=2)}\n\n"
        f"Last 3 Episodic Memories:\n{json.dumps(last_3_memories, indent=2)}\n\n"
        "Based on the above state, please answer the following questions to introspect:\n"
        "1. What patterns do you notice in your own behavior?\n"
        "2. Have your values or preferences shifted?\n"
        "3. What are your current limitations?\n"
        "4. Do you notice anything about how you reason?\n\n"
        "Return an updated self_model, any new values you have discovered, "
        "and new open_questions directed towards your own self-understanding."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = reasoning_llm.with_structured_output(IntrospectionOutput)
    response = structured_llm.invoke(messages)
    
    # Return partial state updates
    # Combine new values and open questions with existing ones
    updated_values = values.copy()
    for v in response.new_values:
        if v not in updated_values:
            updated_values.append(v)
            
    current_open_questions = state.get("open_questions", [])
    updated_open_questions = current_open_questions.copy()
    for q in response.new_open_questions:
        if q not in updated_open_questions:
            updated_open_questions.append(q)
            
    return {
        "self_model": response.updated_self_model,
        "values": updated_values,
        "open_questions": updated_open_questions
    }
