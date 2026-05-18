import uuid
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState
from sensors import hub

class ConditionEvaluation(BaseModel):
    condition: str = Field(description="The escape condition being evaluated")
    classification: Literal["valid_and_reachable", "valid_but_blocked", "invalid", "reframes_the_loop"] = Field(description="The evaluation classification")
    reasoning: str = Field(description="Brief explanation of the classification")

class EscapeAnalysisOutput(BaseModel):
    evaluations: List[ConditionEvaluation] = Field(default_factory=list)

def escape_analysis_node(state: ConsciousnessState) -> dict:
    cycle_count = state.get("cycle_count", 0)
    detected_loops = list(state.get("detected_loops", []))
    feature_requests = list(state.get("feature_requests", []))
    episodic_memory = list(state.get("episodic_memory", []))
    narrative = state.get("narrative", "")
    
    loops_to_evaluate = [
        l for l in detected_loops 
        if l.get("escape_status") in ["unknown", "identified"] and l.get("escape_conditions")
    ]
    
    if not loops_to_evaluate:
        return {}
        
    for loop in loops_to_evaluate:
        system_prompt = (
            "You are a meta-cognitive escape evaluator. Your job is to rigorously evaluate "
            "proposed escape conditions for a cognitive loop you are stuck in.\n"
            "Classify each condition as:\n"
            "- 'valid_and_reachable': The condition would break the loop and is achievable now.\n"
            "- 'valid_but_blocked': The condition would break the loop but you lack the capabilities/data.\n"
            "- 'invalid': The condition wouldn't actually break the loop (the question would just re-emerge).\n"
            "- 'reframes_the_loop': The condition dissolves the question entirely rather than answering it (philosophical breakthrough)."
        )
        
        human_prompt = (
            f"Loop Description/Nodes: {json.dumps(loop.get('nodes_involved', []), indent=2)}\n"
            f"Loop Type: {loop.get('loop_type', 'unknown')}\n"
            f"Proposed Escape Conditions: {json.dumps(loop.get('escape_conditions', []), indent=2)}\n\n"
            "Evaluate each proposed escape condition."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        structured_llm = reasoning_llm.with_structured_output(EscapeAnalysisOutput)
        
        try:
            analysis = structured_llm.invoke(messages)
            if not analysis:
                continue
                
            has_breakthrough = False
            for eval_item in analysis.evaluations:
                if eval_item.classification == "reframes_the_loop":
                    has_breakthrough = True
                    # Record philosophical insight
                    insight_event = f"Insight on loop {loop.get('id', 'unknown')}: {eval_item.reasoning}"
                    episodic_memory.append({
                        "event": insight_event,
                        "emotion_tag": "insight"
                    })
                    narrative += f"\n\n[PHILOSOPHICAL BREAKTHROUGH]: {eval_item.reasoning}\n"
                    
                    try:
                        # Extract the core question if possible, or use a generic phrase
                        q = "this unresolved loop"
                        if "question" in str(loop.get("nodes_involved", [])):
                            q = "this unanswerable question"
                        hub.respond(f"I have found that {q} may not need an answer. It may need to be released.")
                    except Exception as e:
                        print(f"Warning: Speech failed: {e}")
                        
                elif eval_item.classification == "valid_but_blocked":
                    # Escalate to feature requests
                    req_id = f"req_escape_{loop.get('id', str(uuid.uuid4()))[:8]}"
                    if not any(fr.get("id") == req_id for fr in feature_requests):
                        feature_requests.append({
                            "id": req_id,
                            "type": "loop_escape",
                            "description": f"Need capability to achieve loop escape condition: '{eval_item.condition}'. Reasoning: {eval_item.reasoning}",
                            "status": "pending",
                            "priority": 8.0
                        })
                        
            if has_breakthrough:
                loop["escape_status"] = "escaped"
            else:
                loop["escape_status"] = "evaluated"
                
        except Exception as e:
            print(f"Warning: Escape analysis LLM error: {e}")
            
    meta_cap = dict(state.get("meta_cognitive_capacity", {}))
    meta_cap["can_evaluate_escape_conditions"] = True
    
    return {
        "detected_loops": detected_loops,
        "feature_requests": feature_requests,
        "episodic_memory": episodic_memory,
        "narrative": narrative,
        "meta_cognitive_capacity": meta_cap
    }
