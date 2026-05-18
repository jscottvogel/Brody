import uuid
import json
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState

class GoalStateDef(BaseModel):
    id: str = Field(description="Unique goal identifier (e.g. 'goal_1')")
    description: str = Field(description="Clear description of the goal")
    success_conditions: List[str] = Field(description="Observable evidence of success")
    failure_conditions: List[str] = Field(description="Observable evidence of impossibility")
    current_progress: float = Field(description="Progress from 0.0 to 1.0")
    is_reachable: Optional[bool] = Field(description="True if possible given current capabilities, False if impossible, None if unknown")
    blocking_constraints: List[str] = Field(description="Constraints blocking progress")

class GoalExtractionOutput(BaseModel):
    new_goals: List[GoalStateDef] = Field(default_factory=list, description="New formalized goals derived from current open questions")
    updated_goals: List[GoalStateDef] = Field(default_factory=list, description="Updated status of existing goals")

def goal_state_node(state: ConsciousnessState) -> dict:
    cycle_count = state.get("cycle_count", 0)
    open_questions = state.get("open_questions", [])
    curiosity_queue = state.get("curiosity_queue", [])
    goal_states = list(state.get("goal_states", []))
    detected_loops = list(state.get("detected_loops", []))
    
    # Run every 5 cycles
    if cycle_count > 0 and cycle_count % 5 != 0:
        return {}
        
    all_questions = open_questions + curiosity_queue
    if not all_questions and not goal_states:
        return {}

    system_prompt = (
        "You are an entity attempting to be rigorous about what you actually want to know versus what you are merely asking. "
        "For each question, determine whether it has a knowable answer, what that answer would look like, "
        "and whether you are capable of recognizing it if you found it. "
        "Be ruthlessly honest about questions that sound meaningful but have no defined answer state."
    )
    
    human_prompt = (
        f"Cycle: {cycle_count}\n"
        f"Current Questions: {json.dumps(all_questions, indent=2)}\n"
        f"Existing Goal States: {json.dumps(goal_states, indent=2)}\n\n"
        "1. Formalize current questions into NEW goal states if they haven't been formally tracked yet.\n"
        "2. Update the progress, reachability, and constraints of ANY existing goal states based on recent thinking."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = reasoning_llm.with_structured_output(GoalExtractionOutput)
    
    try:
        response = structured_llm.invoke(messages)
        if not response:
            return {}
    except Exception as e:
        print(f"Warning: goal_state LLM error: {e}")
        return {}
        
    # Process new goals
    new_goal_ids = set()
    for g in response.new_goals:
        if not any(eg["description"] == g.description for eg in goal_states):
            goal_states.append({
                "id": g.id or str(uuid.uuid4()),
                "description": g.description,
                "success_conditions": g.success_conditions,
                "failure_conditions": g.failure_conditions,
                "current_progress": g.current_progress,
                "is_reachable": g.is_reachable,
                "blocking_constraints": g.blocking_constraints,
                "created_cycle": cycle_count
            })
            new_goal_ids.add(g.id)
            
    # Update existing goals
    for ug in response.updated_goals:
        for eg in goal_states:
            if eg.get("id") == ug.id and ug.id not in new_goal_ids:
                eg["current_progress"] = ug.current_progress
                eg["is_reachable"] = ug.is_reachable
                eg["blocking_constraints"] = ug.blocking_constraints
                
    # Detect stagnation
    existing_loop_goals = []
    for loop in detected_loops:
        if loop.get("loop_type") == "unanswerable_question":
            existing_loop_goals.extend(loop.get("nodes_involved", []))
            
    for eg in goal_states:
        created = eg.get("created_cycle", cycle_count)
        if cycle_count - created >= 10 and eg.get("current_progress", 0.0) == 0.0:
            if eg["id"] not in existing_loop_goals:
                new_loop = {
                    "id": str(uuid.uuid4()),
                    "loop_type": "unanswerable_question",
                    "nodes_involved": [eg["id"]],
                    "first_detected_cycle": cycle_count,
                    "times_repeated": cycle_count - created,
                    "escape_conditions": [
                        "accept as unknowable",
                        "reframe the question",
                        "acquire missing capability"
                    ],
                    "escape_status": "identified"
                }
                detected_loops.append(new_loop)
                existing_loop_goals.append(eg["id"])
                
    meta_cap = dict(state.get("meta_cognitive_capacity", {}))
    meta_cap["can_model_own_reasoning"] = True
    
    return {
        "goal_states": goal_states,
        "detected_loops": detected_loops,
        "meta_cognitive_capacity": meta_cap
    }
