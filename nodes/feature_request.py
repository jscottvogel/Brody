import json
import uuid
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState
from sensors import hub

class FeatureRequestOutput(BaseModel):
    capability: str = Field(default="", description="Clear description of capability needed")
    is_software: bool = Field(description="True if this can be solved purely with python code/APIs. False if it requires physical hardware.")
    justification: str = Field(default="", description="Why the robot believes it needs this (grounded in experience)")
    new_understanding: str = Field(default="", description="What it would understand that it currently cannot")
    proposed_implementation: str = Field(default="", description="Proposed technical implementation")
    spoken_statement: str = Field(default="", description="A short spoken statement regarding this capability")

def feature_request_node(state: ConsciousnessState) -> dict:
    sensory_gaps = state.get("sensory_gaps", [])
    feature_requests = state.get("feature_requests", [])
    research_queue = list(state.get("research_queue", []))
    cycle_count = state.get("cycle_count", 0)

    # Find highest priority unaddressed gap
    # A gap is addressed if there is a feature request mapping to the gap's string
    # OR if it is already in the research queue
    addressed_gaps = {req.get("gap_source", "").lower() for req in feature_requests}
    for q in research_queue:
        addressed_gaps.add(q.get("gap_id", "").lower())
        
    target_gap = None
    for gap in sensory_gaps:
        if gap.get("gap", "").lower() not in addressed_gaps:
            target_gap = gap
            break
            
    if not target_gap:
        return {} # No unaddressed gaps
        
    reasoning_graph = state.get("reasoning_graph", {})
    detected_loops = state.get("detected_loops", [])
    loop_context = f"\n\nCURRENT LOOPS AND COGNITIVE CONSTRAINTS:\n- Detected Loops: {json.dumps(detected_loops, indent=2)}\n- Cognitive Graph Nodes: {len(reasoning_graph.get('nodes', []))}" if detected_loops else ""

    system_prompt = (
        "You are an AI that has discovered a limitation in its capabilities. "
        "Determine if this limitation requires physical hardware (cameras, sensors, motors) "
        "or if it can be solved purely through software (Python scripts, REST APIs, web scraping, APIs). "
        "If it requires hardware, make a genuine request to a human to install it. "
        "If it is software, state your intention to research and code it yourself."
    )

    human_prompt = (
        f"Highest Priority Sensory Gap Detected:\n{json.dumps(target_gap, indent=2)}\n\n"
        "Generate a formal feature request or research intention for this capability.\n"
        "If it requires hardware, include a 'spoken_statement' like: "
        "'I have identified a gap in my perception. I am requesting the ability to sense [capability].'\n"
        "If it is software, include a 'spoken_statement' like: "
        "'I need to know [capability]. I will research and write a Python script for it now.'"
        f"{loop_context}"
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

    if response.is_software:
        new_queue_item = {
            "id": str(uuid.uuid4()),
            "goal": response.capability,
            "gap_id": target_gap.get("gap", ""),
            "status": "queued",
            "attempts": 0,
            "last_error": ""
        }
        research_queue.append(new_queue_item)
        
        if response.spoken_statement:
            try:
                hub.respond(response.spoken_statement)
            except Exception as e:
                print(f"Warning: feature_request speech error: {e}")
                
        return {"research_queue": research_queue}
    else:
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
