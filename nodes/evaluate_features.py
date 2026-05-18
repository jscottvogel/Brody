import json
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import analytical_llm
from state import ConsciousnessState
from sensors.sensor_hub import hub

class SatisfiedFeature(BaseModel):
    request_id: str = Field(description="The ID of the feature request that was satisfied")
    evidence: str = Field(description="The sensory evidence that proves it was satisfied")
    new_capability_key: str = Field(description="A short snake_case key for the new capability")
    new_capability_description: str = Field(description="A description of the new capability for the capability map")
    spoken_acknowledgment: str = Field(description="A spoken announcement discovering the new capability")

class EvaluateFeaturesOutput(BaseModel):
    satisfied_features: List[SatisfiedFeature] = Field(default_factory=list, description="List of pending features that have been satisfied")

def evaluate_features_node(state: ConsciousnessState) -> dict:
    feature_requests = state.get("feature_requests", [])
    pending_requests = [r for r in feature_requests if r.get("status") == "pending"]
    
    if not pending_requests:
        return {}
        
    sensor_input = state.get("sensor_input", "")
    capability_map = state.get("capability_map", {})
    
    system_prompt = (
        "You are an AI meta-cognitive evaluator. Review the list of pending feature requests "
        "against the current raw sensor input. Determine if there is concrete sensory evidence "
        "that any of these requests have been fulfilled.\n"
        "If fulfilled, provide the request ID, the sensory evidence, a new capability key/description, "
        "and a 'spoken_acknowledgment' formatted like: "
        "'I perceive that my request for [capability] has been fulfilled. I can now [action].'"
    )
    
    human_prompt = (
        f"Pending Feature Requests:\n{json.dumps(pending_requests, indent=2)}\n\n"
        f"Current Capability Map:\n{json.dumps(capability_map, indent=2)}\n\n"
        f"Current Sensor Input:\n{sensor_input}\n\n"
        "Are any pending requests demonstrably satisfied by this sensor input?"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = analytical_llm.with_structured_output(EvaluateFeaturesOutput)
    
    try:
        response = structured_llm.invoke(messages)
        if not response or not response.satisfied_features:
            return {}
    except Exception as e:
        print(f"Warning: evaluate_features LLM error: {e}")
        return {}
        
    updated_requests = feature_requests.copy()
    updated_capability_map = capability_map.copy()
    updated_implemented = state.get("implemented_features", []).copy()
    
    for satisfied in response.satisfied_features:
        # Update request status
        for req in updated_requests:
            if req.get("id") == satisfied.request_id:
                req["status"] = "implemented"
                break
                
        # Update capability map
        updated_capability_map[satisfied.new_capability_key] = satisfied.new_capability_description
        
        # Add to implemented features list
        if satisfied.new_capability_key not in updated_implemented:
            updated_implemented.append(satisfied.new_capability_key)
            
        # Speak acknowledgment
        if satisfied.spoken_acknowledgment:
            try:
                hub.respond(satisfied.spoken_acknowledgment)
            except Exception as e:
                print(f"Warning: Failed to speak feature acknowledgment: {e}")
                
    return {
        "feature_requests": updated_requests,
        "capability_map": updated_capability_map,
        "implemented_features": updated_implemented
    }
