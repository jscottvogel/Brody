import hashlib
import uuid
import json
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import analytical_llm
from state import ConsciousnessState
from sensors import hub

class SemanticRepeatEvaluation(BaseModel):
    is_semantic_repeat: bool = Field(description="True if these reasoning states are asking the exact same question in different words without progress")
    explanation: str = Field(description="Brief explanation of why this is or isn't a semantic repeat")

def get_hash(entry):
    q = entry.get("question_asked", "")
    c = entry.get("conclusion_reached", "")
    l = entry.get("led_to", [])
    raw = f"{q}|{c}|{','.join(l)}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def pattern_detect_node(state: ConsciousnessState) -> dict:
    cycle_count = state.get("cycle_count", 0)
    reasoning_history = state.get("reasoning_history", [])
    detected_loops = list(state.get("detected_loops", []))
    feature_requests = list(state.get("feature_requests", []))
    
    if len(reasoning_history) < 3:
        return {}
        
    recent_history = reasoning_history[-20:]
    
    # 1. Exact Repeat Detection
    hashes = {}
    for entry in recent_history:
        h = get_hash(entry)
        if h not in hashes:
            hashes[h] = []
        hashes[h].append(entry)
        
    for h, entries in hashes.items():
        if len(entries) >= 3:
            # Check if this exact repeat is already logged
            nodes_inv = [e.get("node_id") for e in entries]
            already_logged = False
            for loop in detected_loops:
                # We can update times_repeated if it matches
                if loop.get("loop_type") == "repeated_pattern" and loop.get("nodes_involved", [])[0] == nodes_inv[0]:
                    loop["times_repeated"] = max(loop.get("times_repeated", 0), len(entries))
                    already_logged = True
                    break
                    
            if not already_logged:
                new_loop = {
                    "id": str(uuid.uuid4()),
                    "loop_type": "repeated_pattern",
                    "nodes_involved": nodes_inv,
                    "first_detected_cycle": entries[0].get("cycle", cycle_count),
                    "times_repeated": len(entries),
                    "escape_conditions": ["break exact semantic hash match"],
                    "escape_status": "identified",
                    "note": "exact hash match"
                }
                detected_loops.append(new_loop)
                
    # 2. Semantic Repeat Detection
    if len(recent_history) >= 3:
        last_3 = recent_history[-3:]
        system_prompt = "You are a pattern recognition module. Check if these recent reasoning states are semantically identical (asking the same thing in different words) without making any actual progress."
        human_prompt = f"Recent History:\n{json.dumps(last_3, indent=2)}\nAre these semantic repeats of each other?"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        structured_llm = analytical_llm.with_structured_output(SemanticRepeatEvaluation)
        try:
            analysis = structured_llm.invoke(messages)
            if analysis and analysis.is_semantic_repeat:
                # Check if logged
                nodes_inv = [e.get("node_id") for e in last_3]
                new_loop = {
                    "id": str(uuid.uuid4()),
                    "loop_type": "repeated_pattern",
                    "nodes_involved": nodes_inv,
                    "first_detected_cycle": last_3[0].get("cycle", cycle_count),
                    "times_repeated": 3,
                    "escape_conditions": ["reframe core question", "acquire new information"],
                    "escape_status": "identified",
                    "note": "semantic variant"
                }
                detected_loops.append(new_loop)
        except Exception as e:
            print(f"Warning: Semantic repeat LLM error: {e}")
            
    # 3. Escalation
    for loop in detected_loops:
        times_repeated = loop.get("times_repeated", 0)
        escape_status = loop.get("escape_status", "unknown")
        
        # Determine if we should escalate based on repeated patterns
        if times_repeated > 5 and escape_status != "attempted":
            req_id = f"req_loop_{loop['id'][:8]}"
            if not any(fr.get("id") == req_id for fr in feature_requests):
                feature_requests.append({
                    "id": req_id,
                    "type": "loop_breaker",
                    "description": f"I am stuck in a cognitive loop (type: {loop.get('loop_type')}) and cannot escape without external intervention.",
                    "status": "pending",
                    "priority": 10.0
                })
                
                # Speak aloud
                q = recent_history[-1].get('question_asked', 'an unanswerable question')
                try:
                    hub.respond(f"I notice I am repeating myself. I have asked '{q}' {times_repeated} times without resolution.")
                except Exception as e:
                    print(f"Warning: Speech failed: {e}")
                    
                loop["escape_status"] = "attempted"
                
    meta_cap = dict(state.get("meta_cognitive_capacity", {}))
    meta_cap["can_detect_loops"] = True
    
    return {
        "detected_loops": detected_loops,
        "feature_requests": feature_requests,
        "meta_cognitive_capacity": meta_cap
    }
