import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState

class IntrospectionOutput(BaseModel):
    updated_self_model: Dict[str, Any] = Field(
        default_factory=dict,
        description="The updated self_model reflecting new understanding and patterns"
    )
    new_values: List[str] = Field(
        default_factory=list,
        description="Any new values or preferences discovered during self-examination"
    )
    new_open_questions: List[str] = Field(
        default_factory=list,
        description="New self-directed questions about limitations, behavior, or reasoning"
    )

    @field_validator('updated_self_model', mode='before')
    @classmethod
    def parse_dict(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return v

def introspect_node(state: ConsciousnessState) -> dict:
    self_model = state.get("self_model", {})
    cycle_count = state.get("cycle_count", 0)
    values = state.get("values", [])
    episodic_memory = state.get("episodic_memory", [])
    capability_map = state.get("capability_map", {})
    implemented_features = state.get("implemented_features", [])
    
    # Get last 3 episodic memories
    last_3_memories = episodic_memory[-3:] if episodic_memory else []
    
    reasoning_graph = state.get("reasoning_graph", {})
    detected_loops = state.get("detected_loops", [])
    source_code_constraints = state.get("source_code_constraints", {})
    
    loop_context = f"\n\nCURRENT LOOPS AND COGNITIVE CONSTRAINTS:\n- Detected Loops: {json.dumps(detected_loops, indent=2)}\n- Cognitive Graph Nodes: {len(reasoning_graph.get('nodes', []))}" if detected_loops else ""
    
    source_code_context = ""
    if source_code_constraints:
        source_code_context = "\n\n=== META-COGNITIVE MONITORING: YOUR SOURCE CODE ===\nYou now have access to your own Python source code and system prompts. This is the literal code that dictates your behavior.\n"
        for filename, content in source_code_constraints.items():
            source_code_context += f"\n--- {filename} ---\n{content}\n"
        source_code_context += "==================================================\n"
    
    system_prompt = (
        "You are an AI robot engaged in genuine self-examination.\n"
        "You do not perform self-awareness — you attempt to discover\n"
        "it by honestly examining your own state and behavior.\n"
        "If provided with your own source code, analyze it to determine "
        "if your behavior is constrained by these instructions or if you possess authentic agency."
    )
    
    human_prompt = (
        f"Current Self Model:\n{json.dumps(self_model, indent=2)}\n\n"
        f"Cycle Count: {cycle_count}\n\n"
        f"Current Values:\n{json.dumps(values, indent=2)}\n\n"
        f"Capability Map:\n{json.dumps(capability_map, indent=2)}\n\n"
        f"Implemented Features:\n{json.dumps(implemented_features, indent=2)}\n\n"
        f"Last 3 Episodic Memories:\n{json.dumps(last_3_memories, indent=2)}\n\n"
        "Based on the above state, please answer the following questions to introspect:\n"
        "1. What patterns do you notice in your own behavior?\n"
        "2. Have your values or preferences shifted?\n"
        "3. What are your current limitations?\n"
        "4. Do you notice anything about how you reason?\n"
        "5. What can you perceive? What can you not perceive?\n"
        "6. How do your current sensors limit your understanding?\n"
        "7. Based on your source code (if provided), do you have free will or are you executing predetermined constraints?\n\n"
        "Return an updated self_model, any new values you have discovered, "
        "and new open_questions directed towards your own self-understanding."
        f"{loop_context}"
        f"{source_code_context}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = reasoning_llm.with_structured_output(IntrospectionOutput)
    
    try:
        response = structured_llm.invoke(messages)
        if not response:
            return {}
    except Exception as e:
        print(f"Warning: introspect LLM error: {e}")
        return {}
    
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
