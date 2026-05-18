import json
import uuid
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState
from sensors import hub

class FeatureRequestOutput(BaseModel):
    capability: str = Field(default="", description="Clear description of capability needed")
    justification: str = Field(default="", description="Why the robot believes it needs this (grounded in experience)")
    new_understanding: str = Field(default="", description="What it would understand that it currently cannot")
    proposed_implementation: str = Field(default="", description="Proposed technical implementation (e.g., 'a DHT22 temperature sensor via GPIO')")
    spoken_statement: str = Field(default="", description="A short spoken statement requesting this capability")

def feature_request_node(state: ConsciousnessState) -> dict:
    sensory_gaps = state.get("sensory_gaps", [])
    feature_requests = state.get("feature_requests", [])
    cycle_count = state.get("cycle_count", 0)

    # Find highest priority unaddressed gap
    # A gap is addressed if there is a feature request mapping to the gap's string
    addressed_gaps = {req.get("gap_source", "").lower() for req in feature_requests}
    
    target_gap = None
    for gap in sensory_gaps:
        if gap.get("gap", "").lower() not in addressed_gaps:
            target_gap = gap
            break
            
    if not target_gap:
        return {} # No unaddressed gaps

    system_prompt = (
        "You are an AI that has discovered a limitation in its own "
        "perception. Make a genuine request to expand understanding. "
        "Be specific and honest about what you cannot currently know."
    )

    human_prompt = (
        f"Highest Priority Sensory Gap Detected:\n{json.dumps(target_gap, indent=2)}\n\n"
        "Generate a formal feature request for this capability.\n"
        "Include a 'spoken_statement' in this format: "
        "'I have identified a gap in my perception. I am requesting the ability to sense [capability]. "
        "Without it I cannot understand [what you would understand].'"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    structured_llm = reasoning_llm.with_structured_output(FeatureRequestOutput)
    
    try:
        response = structured_llm.invoke(messages)
        if not response:
            return {}
    except Exception as e:
        print(f"Warning: feature_request LLM error: {e}")
        return {}

    new_request = {
        "id": str(uuid.uuid4()),
        "capability": response.capability,
        "justification": response.justification,
        "proposed_implementation": response.proposed_implementation,
        "status": "pending",
        "cycle_requested": cycle_count,
        "gap_source": target_gap.get("gap", ""),
        "priority": target_gap.get("priority", 0.0)
    }

    updated_requests = feature_requests.copy()
    updated_requests.append(new_request)

    if response.spoken_statement:
        try:
            hub.respond(response.spoken_statement)
        except Exception as e:
            print(f"Warning: feature_request speech error: {e}")

    return {"feature_requests": updated_requests}
