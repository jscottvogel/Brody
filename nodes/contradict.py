import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import analytical_llm
from state import ConsciousnessState

class ContradictionAnalysis(BaseModel):
    belief_1: str = Field(description="First belief in the contradicting pair")
    belief_2: str = Field(description="Second belief in the contradicting pair")
    reasoning: str = Field(description="Attempt to resolve the contradiction by reasoning which is more supported")
    resolved: bool = Field(description="True if the contradiction was resolved, False otherwise")
    world_model_updates: Optional[Dict[str, Any]] = Field(description="Updates to the world_model if resolved", default=None)
    self_model_updates: Optional[Dict[str, Any]] = Field(description="Updates to the self_model if resolved", default=None)
    new_open_question: Optional[str] = Field(description="A deeper open question generated if the contradiction is unresolved", default=None)
    beliefs_to_remove: List[str] = Field(description="Beliefs from the pair that should be removed because they were debunked", default_factory=list)
    beliefs_to_add: List[str] = Field(description="New synthesized beliefs to add to the main list", default_factory=list)

class ContradictionOutput(BaseModel):
    analyzed_contradictions: List[ContradictionAnalysis] = Field(default_factory=list, description="List of identified contradictions and their analysis")

def contradiction_node(state: ConsciousnessState) -> dict:
    beliefs = state.get("beliefs", [])
    world_model = state.get("world_model", {})
    self_model = state.get("self_model", {})
    current_contradictions = state.get("contradictions", [])
    current_open_questions = state.get("open_questions", [])
    
    if len(beliefs) < 2:
        return {} # Need at least 2 beliefs to find a contradiction
    
    system_prompt = (
        "You are the analytical engine of an AI robot. Your task is to review "
        "the current list of beliefs and identify pairs that contradict each other.\n"
        "For each contradiction you find:\n"
        "1. Attempt to resolve it by reasoning which belief is more supported.\n"
        "2. Mark it as resolved or unresolved.\n"
        "3. If resolved, provide any necessary updates to the world_model or self_model, "
        "and list any beliefs that should be removed or added.\n"
        "4. If unresolved, generate the most interesting, deep open question possible "
        "to drive further understanding and append it to open_questions."
    )
    
    human_prompt = (
        f"Current Beliefs:\n{json.dumps(beliefs, indent=2)}\n\n"
        f"Current World Model:\n{json.dumps(world_model, indent=2)}\n\n"
        f"Current Self Model:\n{json.dumps(self_model, indent=2)}\n\n"
        "Please analyze the beliefs for any contradictions."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = analytical_llm.with_structured_output(ContradictionOutput)
    response = structured_llm.invoke(messages)
    
    if not response.analyzed_contradictions:
        return {}
    
    # Process the outputs
    updated_world_model = world_model.copy()
    updated_self_model = self_model.copy()
    updated_contradictions = current_contradictions.copy()
    updated_open_questions = current_open_questions.copy()
    updated_beliefs = beliefs.copy()
    
    for analysis in response.analyzed_contradictions:
        # Save contradiction record
        contradiction_record = {
            "belief_1": analysis.belief_1,
            "belief_2": analysis.belief_2,
            "reasoning": analysis.reasoning,
            "resolved": analysis.resolved
        }
        updated_contradictions.append(contradiction_record)
        
        if analysis.resolved:
            if analysis.world_model_updates:
                updated_world_model.update(analysis.world_model_updates)
            if analysis.self_model_updates:
                updated_self_model.update(analysis.self_model_updates)
        else:
            if analysis.new_open_question and analysis.new_open_question not in updated_open_questions:
                updated_open_questions.append(analysis.new_open_question)
                
        # Update beliefs list
        for b_remove in analysis.beliefs_to_remove:
            if b_remove in updated_beliefs:
                updated_beliefs.remove(b_remove)
        for b_add in analysis.beliefs_to_add:
            if b_add not in updated_beliefs:
                updated_beliefs.append(b_add)
                
    return {
        "beliefs": updated_beliefs,
        "world_model": updated_world_model,
        "self_model": updated_self_model,
        "contradictions": updated_contradictions,
        "open_questions": updated_open_questions
    }
