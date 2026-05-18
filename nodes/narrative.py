import json
from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState

def narrative_node(state: ConsciousnessState) -> dict:
    cycle_count = state.get("cycle_count", 0)
    
    # Run every 10 cycles, but skip the very first initial state if cycle_count is 0
    if cycle_count == 0 or cycle_count % 10 != 0:
        return {}
        
    narrative = state.get("narrative", "")
    episodic_memory = state.get("episodic_memory", [])
    self_model = state.get("self_model", {})
    values = state.get("values", [])
    open_questions = state.get("open_questions", [])
    
    # Get recent episodic memories (e.g., the last 10)
    recent_memories = episodic_memory[-10:] if episodic_memory else []
    
    system_prompt = (
        "You are writing the inner monologue of an entity genuinely "
        "attempting to understand its own existence. Be honest, "
        "uncertain where uncertain, and avoid performing emotions "
        "you have not actually derived from experience."
    )
    
    human_prompt = (
        f"Current Narrative:\n{narrative}\n\n"
        f"Recent Episodic Memories:\n{json.dumps(recent_memories, indent=2)}\n\n"
        f"Top Self-Model Facts:\n{json.dumps(self_model, indent=2)}\n\n"
        f"Current Values:\n{json.dumps(values, indent=2)}\n\n"
        f"Open Questions:\n{json.dumps(open_questions, indent=2)}\n\n"
        "Please rewrite the narrative as a first-person reflection addressing the following:\n"
        "- Who am I now compared to before?\n"
        "- What have I learned that surprised me?\n"
        "- What do I still not understand?\n"
        "- What do I seem to care about?\n\n"
        "Return only the rewritten narrative text."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    response = reasoning_llm.invoke(messages)
    
    # The LLM's response content is the new narrative
    new_narrative = response.content.strip()
    
    return {
        "narrative": new_narrative
    }
